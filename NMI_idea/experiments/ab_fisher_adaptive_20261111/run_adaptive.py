"""Run the paired, no-deadline Fisher/adaptive policy experiment.

The runner is intentionally the only module that sees private anatomy and
physical actions.  A checked native transition JSON is loaded once and shared
by every policy arm.  For each paired scenario, policies receive the same
public-history acoustic packets (common random numbers are keyed by
``capture_id + visit``), while the private renderer/evaluator remains in the
``privateeval`` side of each event record.

Example smoke invocation with the built-in transition fixture::

    python run_adaptive.py --smoke --index 0 --mode constrained \
        --count 2 --output /tmp/fisher-smoke

Native execution is not performed here.  The normal invocation consumes the
deterministic checked-transition artifact produced by the native worker::

    python run_adaptive.py --transitions /path/fisher_native.json \
        --index 0 --mode constrained --count 128 --output /path/result

There is no task-time deadline.  The two diagnostic and two terminal caps are
finite protocol caps and are reported separately from ``no_valid_scan`` and
public false stops.  Time is logged and charged only through the shared soft
cost (default 0.04/s).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
SIBLING = HERE.parent / "ab_closed_loop_reviewed_20261111"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(SIBLING) not in sys.path:
    sys.path.insert(0, str(SIBLING))

from adaptive_policy import (  # noqa: E402
    AdaptivePlanner,
    DEFAULT_TOP_K,
    DIAGNOSTIC_CAP,
    FALSE_STOP_CONFIDENCE,
    PolicyConfig,
    QUALITY_THRESHOLD,
    TERMINAL_CAP,
    TERMINALS,
    TIME_COST,
    Transition,
    TransitionCatalog,
    _json_safe,
    public_stop,
)

# Corrected sibling rendering/evaluator.  Importing these modules after the
# path setup keeps this runner tied to the reviewed implementation freeze.
from acoustics import (  # noqa: E402
    CENTER_TOLERANCE_M,
    AXIS_TOLERANCE_RAD,
    PRIOR_SD,
    CapturedObservation,
    HiddenFrame,
    ProbeAction,
    PrivateScenario,
    geometry_value,
    make_scenario,
    paired_scenarios,
    render_capture,
    response_design,
    terminal_truth,
)


TEST_SEED = 721503
NOISE_SEED = 721504
DEFAULT_COUNT = 128
METHODS = (
    "A_greedy",
    "fixed_pair",
    "B_original_bank",
    "B_plain_topK",
    "B_logdet",
    "B_fisher_topK",
    "B_common_full",
)
ALL_METHODS = METHODS + ("fixed_pair_shift",)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean(value: Any) -> Any:
    return _json_safe(value)


def _dump(path: Path, payload: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(payload), indent=None if compact else 2,
                               allow_nan=False) + "\n", encoding="utf-8")


def _event_id(case_id: int, mode: str, action_id: str, prefix: Sequence[str]) -> str:
    """Common-random id: capture/action plus visit, never method name."""

    visit = tuple(prefix).count(str(action_id))
    return f"case_{int(case_id):04d}/{mode}/{action_id}/visit_{visit}"


def _state_hash(payload: Mapping[str, Any] | None) -> str | None:
    if payload is None:
        return None
    encoded = json.dumps(_clean(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _action_quality_truth(scenario: PrivateScenario, action: ProbeAction) -> dict[str, Any]:
    """Deterministic wrong-window audit, evaluator-only.

    The planner never calls this function.  The audit uses the old geometry,
    response-mean and quality gates, as requested by the paired analysis.
    """

    # ``_true_geometry_value`` is deliberately imported only by the runner.
    import acoustics as _acoustics  # local import documents the privacy boundary

    geometry, center, axis = _acoustics._true_geometry_value(scenario, action)
    mean_quality = float(geometry + response_design(action) @ np.asarray(scenario.response, dtype=float))
    good = bool(center <= CENTER_TOLERANCE_M and axis <= AXIS_TOLERANCE_RAD
                and mean_quality >= QUALITY_THRESHOLD)
    return {
        "mean_quality": mean_quality, "center_error_m": float(center),
        "axis_error_rad": float(axis), "deterministic_geometry_good": good,
    }


def _public_action_payload(action: ProbeAction) -> dict[str, Any]:
    return _clean(asdict(action))


class AcquisitionCache:
    """Renderer/evaluator owned by the runner, not the public planner."""

    def __init__(self, scenario: PrivateScenario, *, case_id: int, mode: str,
                 noise_seed: int = NOISE_SEED, catalog: TransitionCatalog | None = None):
        self.scenario = scenario
        self.case_id = int(case_id)
        self.mode = str(mode)
        self.noise_seed = int(noise_seed)
        self.catalog = catalog
        self._cache: dict[tuple[str, int, str], tuple[CapturedObservation, dict[str, Any]]] = {}

    def capture(self, transition: Transition, prefix: Sequence[str], *, is_terminal: bool) -> tuple[CapturedObservation, dict[str, Any]]:
        event_id = _event_id(self.case_id, self.mode, transition.action_id, prefix)
        visit = tuple(prefix).count(transition.action_id)
        # Include the physical action signature in the memo key so distinct
        # prefixes share random draws but are not incorrectly forced to share
        # geometry.  The event id itself remains common by capture+visit.
        action_key = json.dumps(_public_action_payload(transition.private_action), sort_keys=True)
        key = (transition.action_id, visit, action_key)
        cached = self._cache.get(key)
        from dataclasses import replace
        if cached is None:
            packet = render_capture(
                self.scenario, transition.private_action, event_id=event_id,
                elapsed_s=float(transition.elapsed_s), seed=self.noise_seed,
                exit_state=str(transition.route_id or "checked"), informative=True,
            )
            # Cache the hidden/public stochastic draw only.  Mechanical
            # elapsed/route metadata is reconstructed below on every use, so
            # a shared random packet cannot leak a different arm's state.
            self._cache[key] = (packet, {})
        else:
            packet, _unused = cached
        public = replace(packet.public, event_id=event_id,
                         elapsed_s=float(transition.elapsed_s),
                         exit_state=str(transition.route_id or "checked"))
        packet = CapturedObservation(public, packet.hidden)
        truth_payload: dict[str, Any] = {
            "event_id": event_id,
            "hidden_kind": str(packet.hidden.kind),
            "present": True,
            "elapsed_s": float(transition.elapsed_s),
            "route_id": transition.route_id,
            "native_entry_state_hash": _state_hash({"prefix": list(prefix)}),
            "native_exit_state_hash": _state_hash(transition.output),
            "private_action": _public_action_payload(transition.private_action),
        }
        if is_terminal:
            truth = terminal_truth(self.scenario, transition.private_action,
                                   event_id=event_id, seed=self.noise_seed,
                                   capture=packet)
            truth_payload.update({
                "target_visible": bool(truth.target_visible),
                "target_identity_correct": bool(truth.target_identity_correct),
                "quality_pass": bool(truth.quality_pass),
                "center_error_m": float(truth.center_error_m),
                "axis_error_rad": float(truth.axis_error_rad),
                "fresh_image": bool(truth.fresh_image),
            })
        # Keep only the stochastic packet in the cache; truth metadata above
        # is intentionally rebuilt for the current transition/prefix.
        self._cache[key] = (packet, {})
        return packet, truth_payload


def _synthetic_catalog(index: int, mode: str) -> TransitionCatalog:
    """Small deterministic checked catalogue for smoke tests only.

    It mirrors the reviewed fixture's public action layout and timings but is
    never used for a test run.  Real runs require the fisher-native artifact.
    """

    main_side = "left" if int(index) % 2 == 0 else "right"
    rows: list[dict[str, Any]] = []
    poses = {
        "side_left": ("window_left", (-0.016, 0.013), +0.07, 0.72),
        "side_right": ("window_right", (+0.016, 0.013), -0.07, 0.72),
        "fork_minus": ("diagnostic_mouth", (0.006, -0.003), -0.075, 0.30),
        "fork_plus": ("diagnostic_mouth", (-0.006, -0.003), +0.075, 0.30),
        "left": ("window_left", (-0.016, 0.018), +0.10, 1.00),
        "right": ("window_right", (+0.016, 0.018), -0.10, 1.00),
    }
    # Public transition times follow the reviewed closed-loop witness.  The
    # synthetic route is only a planner/evaluator smoke target, not a native
    # motion certificate.
    direct = {"left": 2.60, "right": 2.675}
    for action_id in TERMINALS:
        window, xy, yaw, dwell = poses[action_id]
        action = {"window": window, "xy_m": xy, "yaw_rad": yaw, "duration_s": dwell, "kind": "terminal"}
        rows.append({"prefix": [], "action_id": action_id, "valid": True,
                     "duration_s": direct[action_id], "elapsed_s": direct[action_id],
                     "route_id": f"smoke/{action_id}", "public_action": action,
                     "private_action": action, "output": {"prefix": [action_id]}})
    probes = {
        "side_left": 1.53, "side_right": 1.595,
        "fork_minus": 1.225, "fork_plus": 1.225,
    }
    for action_id, duration in probes.items():
        window, xy, yaw, dwell = poses[action_id]
        action = {"window": window, "xy_m": xy, "yaw_rad": yaw, "duration_s": dwell, "kind": "probe"}
        rows.append({"prefix": [], "action_id": action_id, "valid": True,
                     "duration_s": duration, "elapsed_s": duration,
                     "route_id": f"smoke/{action_id}", "public_action": action,
                     "private_action": action, "output": {"prefix": [action_id]}})
        # All single-probe exits expose both terminal windows in this smoke
        # catalogue.  A second probe from any single-probe exit exposes the
        # terminal rows used by the planner's finite look-ahead.
        for terminal in TERMINALS:
            t_window, t_xy, t_yaw, t_dwell = poses[terminal]
            t_action = {"window": t_window, "xy_m": t_xy, "yaw_rad": t_yaw,
                        "duration_s": t_dwell, "kind": "terminal"}
            rows.append({"prefix": [action_id], "action_id": terminal, "valid": True,
                         "duration_s": 1.675 if terminal == ("left" if action_id.endswith("left") else "right") else 3.065,
                         "elapsed_s": duration + (1.675 if terminal == ("left" if action_id.endswith("left") else "right") else 3.065),
                         "route_id": f"smoke/{action_id}/{terminal}", "public_action": t_action,
                         "private_action": t_action, "output": {"prefix": [action_id, terminal]}})
    for first, second in (("fork_minus", "fork_plus"), ("fork_plus", "fork_minus")):
        frow = next(row for row in rows if row["action_id"] == first and row["prefix"] == [])
        action = next(row for row in rows if row["action_id"] == second and row["prefix"] == [])
        rows.append({"prefix": [first], "action_id": second, "valid": True,
                     "duration_s": 1.225, "elapsed_s": 2.75,
                     "route_id": f"smoke/{first}/{second}",
                     "public_action": action["public_action"], "private_action": action["private_action"],
                     "output": {"prefix": [first, second]}})
        for terminal in TERMINALS:
            trow = next(row for row in rows if row["action_id"] == terminal and row["prefix"] == [])
            rows.append({"prefix": [first, second], "action_id": terminal, "valid": True,
                         "duration_s": 2.55 if terminal == "left" else 3.065,
                         "elapsed_s": 2.75 + (2.55 if terminal == "left" else 3.065),
                         "route_id": f"smoke/{first}/{second}/{terminal}",
                         "public_action": trow["public_action"], "private_action": trow["private_action"],
                         "output": {"prefix": [first, second, terminal]}})
    return TransitionCatalog(rows_to_transitions(rows), {
        "schema": "fisher_native_transition_smoke_v1",
        "index": int(index), "mode": mode,
        "fixed_pair": ["fork_plus", "fork_minus"] if main_side == "left" else ["fork_minus", "fork_plus"],
        "original_candidate_ids": list(probes),
        "smoke": True,
    })


def rows_to_transitions(rows: Sequence[Mapping[str, Any]]) -> list[Transition]:
    """Use the same normaliser as worker JSON while keeping smoke local."""

    # Importing the private helper would make it part of the public contract;
    # construct through a tiny in-memory JSON round-trip instead.
    from adaptive_policy import _row_to_transition
    return [_row_to_transition(row) for row in rows]


def _load_catalog(args: argparse.Namespace) -> tuple[TransitionCatalog, str | None]:
    if args.smoke:
        return _synthetic_catalog(args.index, args.mode), None
    if args.transitions is None:
        raise SystemExit("--transitions is required unless --smoke is supplied")
    path = Path(args.transitions)
    if path.is_dir():
        names = [
            path / f"fixture_{int(args.index):02d}_{args.mode}.json",
            path / f"fixture_{int(args.index)}_{args.mode}.json",
            path / f"transitions_{int(args.index):02d}_{args.mode}.json",
        ]
        found = next((item for item in names if item.exists()), None)
        if found is None:
            matches = sorted(path.glob("*.json"))
            found = matches[0] if len(matches) == 1 else None
        if found is None:
            raise FileNotFoundError(f"no transition JSON for index={args.index}, mode={args.mode} under {path}")
        path = found
    return TransitionCatalog.from_json(path, index=args.index, mode=args.mode), str(path)


def _source_manifest(transition_path: str | None) -> dict[str, Any]:
    names = ("adaptive_policy.py", "run_adaptive.py", "forecast_fast.py")
    names_sibling = ("acoustics.py", "planning.py", "run_closed_loop.py", "fixture.py")
    result = {name: {"path": str(HERE / name), "sha256": _sha(HERE / name)} for name in names}
    result.update({f"reviewed/{name}": {"path": str(SIBLING / name), "sha256": _sha(SIBLING / name)}
                   for name in names_sibling})
    if transition_path:
        path = Path(transition_path)
        result["native_transition_json"] = {"path": str(path), "sha256": _sha(path)}
    return result


def _make_public_event_payload(transition: Transition, packet: Any) -> dict[str, Any]:
    return {
        "action_id": transition.action_id,
        "public_action": _public_action_payload(transition.public_action),
        "public": _clean(asdict(packet.public)),
        "route_id": transition.route_id,
        "duration_s": float(transition.duration_s),
        "elapsed_s": float(transition.elapsed_s),
        "valid_motion": bool(transition.valid),
        "output_prefix": list(transition.next_prefix),
    }


def _terminal_forecast(planner: AdaptivePlanner, transition: Transition) -> float:
    return float(__import__("acoustics").scan_probability(
        transition.public_action, planner.memory,
        geometry_value_=geometry_value(transition.public_action), informative=True))


def _evaluate_wrong_window(catalog: TransitionCatalog, scenario: PrivateScenario,
                           event: Mapping[str, Any]) -> dict[str, Any]:
    """Post-hoc wrong-window metric; never returned to the planner."""

    prefix = tuple(event.get("prefix_before", ()))
    selected_id = str(event.get("action_id"))
    selected_transition = catalog.get(prefix, selected_id)
    if selected_transition is None:
        return {"status": "unavailable", "selected": None, "alternate": None}
    selected = _action_quality_truth(scenario, selected_transition.private_action)
    other_id = "right" if selected_id == "left" else "left"
    alternate_transition = catalog.get(prefix, other_id)
    if alternate_transition is None or not alternate_transition.valid:
        return {"status": "alternate_unavailable", "selected": selected,
                "alternate": None, "wrong_window": False,
                "selected_good": selected["deterministic_geometry_good"],
                "alternate_good": None}
    alternate = _action_quality_truth(scenario, alternate_transition.private_action)
    selected_good = bool(selected["deterministic_geometry_good"])
    alternate_good = bool(alternate["deterministic_geometry_good"])
    return {
        "status": "compared", "selected": selected, "alternate": alternate,
        "selected_good": selected_good, "alternate_good": alternate_good,
        "wrong_window": bool(not selected_good and alternate_good),
        "two_good": bool(selected_good and alternate_good),
        "two_bad": bool(not selected_good and not alternate_good),
    }


def _run_episode(catalog: TransitionCatalog, scenario: PrivateScenario, *, case_id: int,
                 mode: str, config: PolicyConfig, acquisition: AcquisitionCache) -> dict[str, Any]:
    private_catalog = getattr(acquisition, "catalog", None)
    planner = AdaptivePlanner(catalog, config)
    prefix: tuple[str, ...] = ()
    events: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    first_scan_prefix: list[str] | None = None
    first_scan_action: str | None = None
    first_success_s: float | None = None
    false_stop = False
    success = False
    reason = "undecided"
    no_valid_scan = False
    diagnostic_cap_hit = False
    terminal_cap_hit = False
    wrong_window_records: list[dict[str, Any]] = []
    wall_planning_s = 0.0
    wall_decision_s = 0.0
    wall_assimilation_s = 0.0
    decision_call_count = 0

    while True:
        if planner.terminal_count >= TERMINAL_CAP:
            terminal_cap_hit = True
            reason = "terminal_cap_hit"
            break
        started = time.perf_counter()
        decision = planner.choose(prefix)
        decision_elapsed = time.perf_counter() - started
        wall_planning_s += decision_elapsed
        wall_decision_s += decision_elapsed
        decision_call_count += 1
        decisions.append(decision.as_dict())
        if decision.kind in {"no_valid_scan", "exit"} or decision.action_id is None:
            no_valid_scan = not bool(catalog.options(prefix, kind="terminal"))
            diagnostic_cap_hit = bool(planner.diagnostic_count >= DIAGNOSTIC_CAP)
            reason = "no_valid_scan" if no_valid_scan else ("diagnostic_cap_hit" if diagnostic_cap_hit else decision.reason)
            break
        transition = catalog.get(prefix, decision.action_id)
        private_transition = (private_catalog.get(prefix, decision.action_id)
                              if private_catalog is not None else transition)
        if transition is None or private_transition is None or not transition.valid or not private_transition.valid:
            no_valid_scan = True
            reason = "no_valid_scan"
            events.append({
                "kind": "invalid_transition", "action_id": decision.action_id,
                "prefix_before": list(prefix), "decision": decision.as_dict(),
            })
            break
        terminal = transition.kind == "terminal" or decision.kind == "commit"
        if terminal and first_scan_prefix is None:
            first_scan_prefix, first_scan_action = list(prefix), transition.action_id
        packet, privateeval = acquisition.capture(private_transition, prefix, is_terminal=terminal)
        event = _make_public_event_payload(transition, packet)
        event.update({
            "kind": "terminal" if terminal else "diagnostic",
            "prefix_before": list(prefix),
            "decision": decision.as_dict(),
            "forecast_success_probability": _terminal_forecast(planner, transition) if terminal else None,
            "forecast_log_score": None,
            "forecast_brier": None,
            "privateeval": privateeval,
        })
        planner_public = packet.public
        if terminal:
            forecast = float(event["forecast_success_probability"])
            # Truth is only scored after the public decision/action is fixed.
            accepted = bool(privateeval.get("quality_pass", False))
            event["forecast_log_score"] = float(-math.log(max(forecast if accepted else 1.0 - forecast, 1.0e-12)))
            event["forecast_brier"] = float((forecast - float(accepted)) ** 2)
        started = time.perf_counter()
        public_diag = planner.observe(transition, planner_public, event_id=str(packet.public.event_id))
        assimilation_elapsed = time.perf_counter() - started
        wall_planning_s += assimilation_elapsed
        wall_assimilation_s += assimilation_elapsed
        event["wall_decision_s"] = decision_elapsed
        event["wall_assimilation_s"] = assimilation_elapsed
        event["planner_observation"] = public_diag
        events.append(event)
        prefix = transition.next_prefix
        if terminal:
            truth_success = bool(privateeval.get("quality_pass", False))
            if truth_success and first_success_s is None:
                first_success_s = float(transition.elapsed_s)
                success = True
            stop = public_stop(planner_public)
            event["public_stop"] = bool(stop)
            event["accepted"] = truth_success
            if stop:
                false_stop = not truth_success
                reason = "stopped_on_target" if truth_success else "false_public_stop"
                break
            if planner.terminal_count >= TERMINAL_CAP:
                terminal_cap_hit = True
                reason = "terminal_cap_hit"
                break
            # Keep the second terminal attempt in the same evaluator.  No
            # deadline and no hidden truth are used in the next decision.
        else:
            if planner.diagnostic_count >= DIAGNOSTIC_CAP:
                # The next choose() call commits if a valid terminal exists;
                # do not mark a cap hit until it truly prevents that action.
                diagnostic_cap_hit = True
        if len(events) > DIAGNOSTIC_CAP + TERMINAL_CAP + 2:
            # Defensive state-machine guard; this is not a time deadline.
            reason = "event_cap_guard"
            break

    terminal_events = [event for event in events if event.get("kind") == "terminal"]
    for event in terminal_events:
        wrong_window_records.append(_evaluate_wrong_window(private_catalog or catalog, scenario, event))
    # A diagnostic cap is a protocol occupancy metric; it is only a terminal
    # reason when no valid terminal was available after the cap.
    if reason == "diagnostic_cap_hit" and catalog.options(prefix, kind="terminal"):
        reason = "diagnostic_cap_hit_before_valid_terminal"
    return {
        "success": bool(success), "first_success_s": first_success_s,
        "first_success_with_planning_s": first_success_s,
        "success_with_planning_charged": bool(success),
        "restricted_time_s": None if first_success_s is None else first_success_s,
        "elapsed_s": float(planner.elapsed_s), "wall_planning_s": float(wall_planning_s),
        "wall_decision_s": float(wall_decision_s),
        "wall_assimilation_s": float(wall_assimilation_s),
        "decision_call_count": int(decision_call_count),
        "value_calls": int(planner.value_calls),
        "fisher_proposal_calls": int(planner.fisher_calls),
        "outcome_tree_calls": int(planner.outcome_tree_calls),
        "first_choice": decisions[0].get("action_id") if decisions else None,
        "terminal_capture_count": int(planner.terminal_count),
        "diagnostic_capture_count": int(planner.diagnostic_count),
        "diagnostic_cap_hit": bool(diagnostic_cap_hit),
        "terminal_cap_hit": bool(terminal_cap_hit),
        "no_valid_scan": bool(no_valid_scan),
        "false_stop": bool(false_stop), "correct_stop": bool(success and reason == "stopped_on_target"),
        "reason": reason, "first_scan_prefix": first_scan_prefix,
        "first_scan_action": first_scan_action,
        "material_belief": planner.memory.snapshot(),
        "observed_logdet_gains": [float(value) for value in planner.observed_logdet],
        "decisions": decisions, "events": events,
        "wrong_window": wrong_window_records,
        "protocol": {
            "diagnostic_cap": DIAGNOSTIC_CAP, "terminal_cap": TERMINAL_CAP,
            "lambda_time": float(config.lambda_time), "time_deadline": None,
            "terminal_truth_used_for_decisions": False,
        },
    }


def run_fixture(catalog: TransitionCatalog, *, index: int, mode: str, count: int,
                output: Path, seed: int = TEST_SEED, noise_seed: int = NOISE_SEED,
                top_k: int = DEFAULT_TOP_K, lambda_time: float = TIME_COST,
                methods: Sequence[str] = METHODS,
                transition_path: str | None = None) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    scenarios = paired_scenarios(seed, int(count))
    configs = [PolicyConfig(method=method, top_k=top_k, lambda_time=lambda_time)
               for method in methods]
    common_manifest = {
        "schema": "fisher_adaptive_paired_v1",
        "fixture_index": int(index), "mode": str(mode),
        "seed": int(seed), "noise_seed": int(noise_seed), "count": int(count),
        "no_deadline": True, "soft_time_cost_per_s": float(lambda_time),
        "diagnostic_cap": DIAGNOSTIC_CAP, "terminal_cap": TERMINAL_CAP,
        "methods": list(methods), "top_k": int(top_k),
        "transition_catalog": catalog.public_manifest(),
        "source_manifest": _source_manifest(transition_path),
        "public_stop": {"confidence": FALSE_STOP_CONFIDENCE, "quality": QUALITY_THRESHOLD},
        "private_truth_policy": "evaluator_only_disjoint_privateeval",
    }
    _dump(output / "FROZEN.json", common_manifest)
    records: list[dict[str, Any]] = []
    public_catalog = catalog.public_view()
    common_cache_entries = 0
    for case_id, scenario in enumerate(scenarios):
        # Deterministic renderer cache is shared across arms; no method name
        # enters event ids, so identical captures/visits use common randoms.
        acquisition = AcquisitionCache(scenario, case_id=case_id, mode=mode,
                                       noise_seed=noise_seed, catalog=catalog)
        for config in configs:
            row = _run_episode(public_catalog, scenario, case_id=case_id, mode=mode,
                               config=config, acquisition=acquisition)
            records.append({"case_id": int(case_id), "arm": config.method,
                            "pair_subject": int(case_id), **row})
        common_cache_entries += len(acquisition._cache)
        if (case_id + 1) % 16 == 0:
            print(f"fixture_{int(index):02d}_{mode}: paired cases {case_id + 1}/{count}", flush=True)
    result = {**common_manifest, "records": records,
              "record_count": len(records),
              "common_random_cache_entries": int(common_cache_entries)}
    _dump(output / "RESULT.json", result, compact=True)
    # JSONL is convenient for the external branch-attribution/CI scripts and
    # preserves raw episode events without requiring a full JSON parse.
    with (output / "EPISODES.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(_clean(row), allow_nan=False, separators=(",", ":")) + "\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=Path, default=None,
                        help="checked Fisher-native transition JSON (required outside --smoke)")
    parser.add_argument("--smoke", action="store_true", help="use deterministic planner-only smoke catalogue")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--mode", choices=("constrained", "open"), default="constrained")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=TEST_SEED)
    parser.add_argument("--noise-seed", type=int, default=NOISE_SEED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--lambda-time", type=float, default=TIME_COST)
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=list(METHODS))
    args = parser.parse_args(argv)
    catalog, transition_path = _load_catalog(args)
    run_fixture(catalog, index=args.index, mode=args.mode, count=args.count,
                output=args.output, seed=args.seed, noise_seed=args.noise_seed,
                top_k=args.top_k, lambda_time=args.lambda_time,
                methods=args.methods,
                transition_path=transition_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

