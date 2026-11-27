"""Run the frozen 2×2 and common-first policies with native motion and synthetic observations."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from acoustics import PRIOR_SD, QUALITY_THRESHOLD, CapturedObservation, HiddenFrame, paired_scenarios, render_capture, terminal_truth
from fixture import MotionRollouts, NativeMotionEnvironment, TERMINALS, make_rig
from planning import (
    COMPUTATION_RESERVE_S, METHODS, PublicModel, action_at, assimilate, choose_continuation,
    choose_initial, new_belief,
)


HERE = Path(__file__).resolve().parent
TEST_SEED = 721419
NOISE_SEED = 721420
TEST_COUNT = 128
MAX_TERMINAL_CAPTURES = 2
FALSE_STOP_CONFIDENCE = 0.65
FREEZE_SOURCES = ("fixture.py", "robot_motion.py", "acoustics.py", "planning.py", "run_closed_loop.py", "PROTOCOL.txt", "IMPLEMENTATION_FREEZE.txt", "ASSOCIATION_CORRECTION.txt")


def json_clean(value):
    if isinstance(value, dict):
        return {k: json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, np.generic):
        return json_clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dump(path, payload, *, compact=False):
    path.write_text(json.dumps(json_clean(payload), indent=None if compact else 2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_signature(motion, state):
    if state.native is None:
        return {"q": motion.native.q_seed.tolist(), "qdot": [0.0] * 8, "snapshot_sha256": None}
    snap = state.native.snapshot
    serialized = json.dumps(json_clean(snap), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"q": state.native.q.tolist(), "qdot": state.native.qdot.tolist(),
            "snapshot_sha256": hashlib.sha256(serialized.encode()).hexdigest()}


def create_retry_catalog(model, largest_deadline):
    """Check feasible second terminal visits offline for all shared first routes."""
    pool = list(model.rollouts.cache.values())
    for item in pool:
        exit_state = item.transfer.output
        if item.capture.kind != "terminal" or not exit_state.valid:
            continue
        remaining = largest_deadline - exit_state.elapsed - COMPUTATION_RESERVE_S
        if remaining <= 0.0:
            continue
        model.scans(exit_state, remaining, predict_unseen=True)
    model.rollouts.finish()


def snapshot_material(rig, belief):
    return {"address": rig.address, **belief.snapshot()}


class Acquisition:
    """The private renderer/evaluator is behind a public packet boundary."""

    def __init__(self, rig, physical_motion, scenario, informative, *, reset=False, all_missing=False):
        self.rig = rig
        self.motion = physical_motion
        self.scenario = scenario
        self.informative = informative
        self.all_missing = all_missing
        reset_noise = np.random.default_rng(np.random.SeedSequence([NOISE_SEED, scenario.seed, 19]))
        self.terminal_scenario = replace(scenario, response=reset_noise.normal(size=2)*PRIOR_SD) if reset else scenario
        self.cache = {}

    def capture(self, state, name):
        key = (*state.key, name)
        if key in self.cache:
            return self.cache[key]
        result = self.motion.capture(state, name)
        output = result.transfer.output
        if not output.valid:
            captured = (result, None, None, None)
        else:
            public_action = action_at(result.capture, output.pose, self.rig.measured)
            physical_action = action_at(result.capture, output.pose, self.rig.actual_surface)
            kind = result.capture.kind
            event = f"{name}/visit_{state.key.count(name)}"
            # No planner-specific input enters rendering or terminal acceptance.
            world = self.terminal_scenario if kind == "terminal" else self.scenario
            packet = render_capture(
                world, physical_action, event_id=event, elapsed_s=output.elapsed,
                seed=NOISE_SEED, exit_state=result.transfer.route_id,
                informative=self.informative or kind == "terminal",
            )
            physical_id = "/".join((*state.key, event))
            public_packet = replace(packet.public, event_id=physical_id)
            if self.all_missing and kind != "terminal":
                public_packet = replace(public_packet, detected=False, confidence=0.0,
                                        quality=float("nan"), candidate_type="missing")
                private_packet = HiddenFrame("missing", float("nan"), float("nan"), float("nan"), False, 0.0)
            else:
                private_packet = packet.hidden
            packet = CapturedObservation(public_packet, private_packet)
            truth = terminal_truth(world, physical_action, event_id=physical_id,
                                   seed=NOISE_SEED, capture=packet) if kind == "terminal" else None
            captured = (result, public_action, packet, truth)
        self.cache[key] = captured
        return captured


def public_stops(packet):
    return bool(packet.detected and packet.confidence >= FALSE_STOP_CONFIDENCE and packet.quality >= QUALITY_THRESHOLD)


def execute_episode(model, acquisitions, method, deadline, initial_choice, initial_wall, *,
                    forced_sequence=(), forced_name=None, read_new=None):
    rig = model.rig
    state = rig.start
    use_new = METHODS[method][0] if read_new is None else read_new
    memory = new_belief()
    admitted_factors = []
    elapsed_compute = initial_wall
    events = []
    first_success_s = None
    first_success_wall_s = None
    false_stop = False
    correct_stop = False
    number_terminal = 0
    previous_terminal = None
    choice = initial_choice.name if initial_choice is not None else forced_name
    planned_value = None if initial_choice is None else initial_choice.value
    agenda = list(forced_sequence if forced_name is not None else initial_choice.sequence)
    fixed_prefix_length = len(forced_sequence)
    prefix_signature = None
    stopped = False
    reason = "no_checked_continuation"
    while len(events) < 4 and not stopped:
        if not agenda:
            if number_terminal >= MAX_TERMINAL_CAPTURES or not state.valid:
                break
            t = time.perf_counter()
            continuation = choose_continuation(
                model, state, memory, deadline - state.elapsed,
                previous_terminal=previous_terminal,
            )
            elapsed_compute += time.perf_counter() - t
            if not continuation.sequence:
                break
            agenda = list(continuation.sequence)
        name = agenda.pop(0)
        predicted = model.rollouts.get(state.key, name)
        if predicted is None or not predicted.transfer.output.valid:
            reason = "no_checked_candidate"
            break
        if state.elapsed + predicted.transfer.duration > deadline + 1e-8:
            reason = "deadline_before_next_capture"
            break
        entered = native_signature(acquisitions.motion, state)
        result, action, packet, truth = acquisitions.capture(state, name)
        state = result.transfer.output
        event_time = state.elapsed
        event = {
            "capture": name, "kind": result.capture.kind, "route_id": result.transfer.route_id,
            "duration_s": result.transfer.duration, "elapsed_s": event_time,
            "model_clock_s": state.elapsed, "wall_charged_clock_s": event_time + elapsed_compute, "state_at_entry": entered,
            "state_at_exit": native_signature(acquisitions.motion, state), "accepted": False,
            "valid_motion": state.valid,
        }
        events.append(event)
        if not state.valid:
            reason = result.transfer.reason
            break
        if event_time > deadline + 1e-8:
            reason = "deadline_at_actual_exit"
            break
        public = packet.public
        is_terminal = result.capture.kind == "terminal"
        event.update({
            "local_executed_action": asdict(action), "public": asdict(public),
            "hidden_kind": packet.hidden.kind,
            "masked_from_policy": not use_new and not is_terminal,
        })
        if (is_terminal or use_new) and method != "A_geometry":
            t = time.perf_counter()
            assimilate(memory, action, public, acquisitions.informative)
            if acquisitions.informative or is_terminal:
                admitted_factors.append((action, public))
            elapsed_compute += time.perf_counter() - t
        if fixed_prefix_length == len(events):
            prefix_signature = native_signature(acquisitions.motion, state)
        if not is_terminal:
            continue
        number_terminal += 1
        previous_terminal = name
        accepted = bool(truth.quality_pass)
        event["accepted"] = accepted
        event["acceptance"] = asdict(truth)
        if accepted and first_success_s is None:
            first_success_s = event_time
            first_success_wall_s = event_time + elapsed_compute
        if public_stops(public):
            stopped = True
            correct_stop = accepted
            false_stop = not accepted
            reason = "stopped_on_target" if accepted else "false_public_stop"
        elif number_terminal >= MAX_TERMINAL_CAPTURES:
            reason = "terminal_retry_limit"
    check = new_belief()
    for action, packet in admitted_factors:
        assimilate(check, action, packet, acquisitions.informative)
    cached_mean, cached_cov = memory.posterior()
    batch_mean, batch_cov = check.posterior()
    memory_error = float(max(np.max(np.abs(cached_mean - batch_mean)), np.max(np.abs(cached_cov - batch_cov))))
    if memory_error > 1e-12:
        raise RuntimeError("address cache differs from replay of the same public factors")
    had_target = False
    loss_seen = False
    reacquired = False
    for event in events:
        kind = event.get("hidden_kind")
        if kind == "target":
            reacquired |= loss_seen
            had_target = True
        elif kind == "missing" and had_target:
            loss_seen = True
    return {
        "success": first_success_s is not None, "first_success_s": first_success_s,
        "first_success_with_planning_s": first_success_wall_s,
        "success_with_planning_charged": first_success_wall_s is not None and first_success_wall_s <= deadline,
        "restricted_time_s": deadline if first_success_s is None else first_success_s,
        "deadline_s": deadline, "wall_planning_s": elapsed_compute, "sim_seconds": state.elapsed,
        "first_choice": choice, "predicted_initial_utility": planned_value,
        "terminal_capture_count": number_terminal,
        "misses": sum(x.get("hidden_kind") == "missing" for x in events),
        "clutter": sum(x.get("hidden_kind") == "distractor" for x in events),
        "loss_then_reacquired": reacquired, "correct_stop": correct_stop, "false_stop": false_stop,
        "same_first": forced_name, "same_first_exit": prefix_signature, "events": events, "reason": reason,
        "material_belief": snapshot_material(rig, memory), "cache_replay_error": memory_error,
    }


def paired_rows(model, physical_motion, cases, variants):
    # Priors at the initial state are identical across hidden objects. Profile
    # each initial decision once per fixture/variant and charge it to every arm.
    preplanned = {}
    for variant, (deadline, informative) in variants.items():
        preplanned[variant] = {}
        for method in METHODS:
            started = time.perf_counter()
            choice = choose_initial(method, model, new_belief(), deadline, informative)
            preplanned[variant][method] = (choice, time.perf_counter() - started)
    rows = []
    for case_id, scenario in enumerate(cases):
        for variant, (deadline, informative) in variants.items():
            acquisitions = Acquisition(model.rig, physical_motion, scenario, informative, reset=variant == "response_reset",
                                       all_missing=variant == "all_missing")
            for method, (choice, wall) in preplanned[variant].items():
                row = execute_episode(model, acquisitions, method, deadline, choice, wall)
                rows.append({"case_id": case_id, "variant": variant, "arm": method, **row})
            if variant not in ("primary", "witness", "no_information", "all_missing"):
                continue
            side = (f"side_{model.rig.main_side}",)
            primary = [("side", side), ("pair", model.fixed_pair)]
            other = [(name, seq) for name, seq in model.rig.programs.items()
                     if seq not in (side, model.fixed_pair)]
            for label, sequence in [*primary, *other]:
                arms = []
                for read in (False, True):
                    row = execute_episode(model, acquisitions, "B_I1_F1", deadline, None, 0.0,
                                          forced_sequence=sequence, forced_name=label, read_new=read)
                    rows.append({"case_id": case_id, "variant": variant, "arm": f"same_{label}_I{int(read)}", **row})
                    arms.append(row)
                if arms[0]["same_first_exit"] != arms[1]["same_first_exit"]:
                    raise RuntimeError("same-first arms have different native exit states")
        if (case_id + 1) % 32 == 0:
            print(model.rig.name, "paired hidden objects completed", case_id + 1, flush=True)
    return rows


def variants_for(mode, model):
    deadline = model.deadline
    if deadline is None:
        return {}
    if mode == "open":
        return {"open": (deadline, True)}
    runs = {
        "primary": (deadline, True), "long1_5": (1.5 * deadline, True),
        "long2": (2.0 * deadline, True), "no_information": (deadline, False),
        "response_reset": (deadline, True), "all_missing": (deadline, False),
    }
    if model.witness["deadline_s"] is not None:
        runs["witness"] = (model.witness["deadline_s"], True)
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--mode", choices=("constrained", "open"), default="constrained")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=TEST_COUNT)
    parser.add_argument("--geometry-only", action="store_true")
    args = parser.parse_args()
    with NativeMotionEnvironment() as native:
        rig = make_rig(args.index, args.mode, native)
        dest = args.output / rig.name
        dest.mkdir(parents=True, exist_ok=True)
        if (dest / "RESULT.json").exists():
            raise FileExistsError(f"result already exists in {dest}; use a fresh output directory")
        model = PublicModel(rig, MotionRollouts(rig, native, dest / "calibration"))
        model.calibrate()
        variants = variants_for(args.mode, model)
        print(rig.name, "public timing", model.public_report(), flush=True)
        if variants:
            create_retry_catalog(model, max(v[0] for v in variants.values()))
        frozen = {
            "fixture": rig.public_metadata(), "planning": model.public_report(), "variants": variants,
            "source_sha256": {p: sha(HERE / p) for p in FREEZE_SOURCES},
            "motion_fingerprint": native.replay._state_fingerprint(),
            "prior_scope": "synthetic coarse patch; all policies share model",
            "anatomy_seed": TEST_SEED, "observation_seed": NOISE_SEED, "planned_anatomy_count": args.count,
        }
        dump(dest / "FROZEN.json", frozen)
        if args.geometry_only:
            return
        physical = MotionRollouts(rig, native, dest / "execution", model=model.rollouts)
        if not variants:
            raise RuntimeError("public engineering has no direct/probe schedule; retained FROZEN.json")
        scenarios = paired_scenarios(TEST_SEED, args.count)
        rows = paired_rows(model, physical, scenarios, variants)
        physical.finish()
        dump(dest / "RESULT.json", {**frozen, "records": rows, "unique_native_execution_routes": len(physical.rows),
                                    "distinct_executed_capture_states": len(physical.cache),
                                    "identical_state_cache_hits": physical.cache_hits}, compact=True)
        print(rig.name, "done", len(rows), "rows", len(physical.rows), "native routes", flush=True)


if __name__ == "__main__":
    main()
