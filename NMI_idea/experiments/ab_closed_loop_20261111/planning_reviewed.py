"""Shared public continuation checks and the closed-loop decision rules."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import time

import numpy as np

from acoustics import (
    MixtureAddressMemory, ProbeAction, PublicObservation, condition, geometry_value,
    predict_outcomes, scan_probability,
)
from fixture import DEADLINE_EPS_S, TERMINALS, local_pose


TIME_COST = 0.04
COMPUTATION_RESERVE_S = 0.020
METHODS = {
    "I0_F0": (False, False), "I0_F1": (False, True),
    "I1_F0": (True, False), "B_I1_F1": (True, True),
    "A_greedy": (True, True), "A_geometry": (True, True), "pair_fixed": (True, True),
}


@dataclass(frozen=True)
class ProbePlan:
    name: str
    sequence: tuple[str, ...]
    valid: bool
    elapsed: float
    exit: object
    captures: tuple


@dataclass(frozen=True)
class Scan:
    name: str
    seconds: float
    action: ProbeAction


@dataclass(frozen=True)
class Choice:
    name: str
    sequence: tuple[str, ...]
    value: float


def action_at(capture, pose, frame):
    pos = local_pose(frame, pose)
    window = f"window_{capture.window}" if capture.window != "fork" else "diagnostic_mouth"
    return ProbeAction(window, tuple(pos[:2]), float(pos[2]), capture.dwell_s,
                       "terminal" if capture.kind == "terminal" else "probe")


def new_belief():
    return MixtureAddressMemory()


def assimilate(belief, action, observation, informative=True):
    if informative or action.kind == "terminal":
        return condition(belief, action, observation, geometry_value_=geometry_value(action))
    return False


class PublicModel:
    def __init__(self, rig, rollouts):
        self.rig = rig
        self.rollouts = rollouts
        self.programs: dict[str, ProbePlan] = {}
        self.initial_scans = ()
        self.deadline = None
        self.witness = None
        self.fixed_pair = self.rig.fixed_pair

    def calibrate(self):
        """Build finite public native witness paths before any hidden world is drawn."""
        t0 = time.perf_counter()
        for terminal in TERMINALS:
            self.rollouts.capture(self.rig.start, terminal)
        self.initial_scans = self.scans(self.rig.start, np.inf)
        for name, seq in self.rig.programs.items():
            exit_state, captures = self.rollouts.program(self.rig.start, seq)
            self.programs[name] = ProbePlan(
                name, seq, exit_state.valid, exit_state.elapsed, exit_state, tuple(captures),
            )
            if exit_state.valid:
                for terminal in TERMINALS:
                    self.rollouts.capture(exit_state, terminal)
        valid_probe = [plan.elapsed for plan in self.programs.values() if plan.valid]
        if self.initial_scans and valid_probe:
            self.deadline = max(scan.seconds for scan in self.initial_scans) + min(valid_probe)
        self.witness = self.witness_deadline()
        complete_pairs = []
        for plan in self.programs.values():
            if not plan.valid or not plan.name.startswith("pair_"):
                continue
            exits = self.scans(plan.exit, np.inf)
            if len(exits) == 2:
                complete_pairs.append((plan.elapsed + max(scan.seconds for scan in exits), plan.name, plan.sequence))
        if complete_pairs:
            self.fixed_pair = min(complete_pairs)[2]
        self.calibration_seconds = time.perf_counter() - t0
        self.rollouts.finish()

    def scans(self, state, remaining, *, predict_unseen=False):
        if not state.valid:
            return ()
        options = []
        for name in TERMINALS:
            predicted = self.rollouts.get(state.key, name)
            if predicted is None and predict_unseen:
                capture = self.rig.captures[name]
                candidates = self.rollouts.candidates(state, capture)
                if not candidates or min(row[2] for row in candidates) + capture.dwell_s > remaining:
                    continue
                predicted = self.rollouts.capture(state, name)
            if predicted is None or not predicted.transfer.output.valid:
                continue
            q_entry = self.rollouts.native.q_seed if state.native is None else state.native.q
            qdot_entry = np.zeros(8) if state.native is None else state.native.qdot
            row = next((row for row in self.rollouts.rows if row["route"] == predicted.transfer.route_id), None)
            if row is None:
                raise RuntimeError("checked continuation is missing its entry trace")
            first = row["steps"][0]
            if not np.allclose(q_entry, first["start_q"], rtol=0, atol=1e-7):
                raise RuntimeError("continuation uses an outdated measured joint state")
            if not np.allclose(qdot_entry, first["start_qdot"], rtol=0, atol=1e-7):
                raise RuntimeError("continuation uses an outdated measured velocity")
            seconds = predicted.transfer.duration
            if seconds <= remaining + DEADLINE_EPS_S:
                action = action_at(predicted.capture, predicted.transfer.output.pose, self.rig.measured)
                options.append(Scan(name, seconds, action))
        return tuple(options)

    def witness_deadline(self):
        central, crossover = {}, {}
        for name, plan in self.programs.items():
            if not plan.valid:
                continue
            times = {scan.name: plan.elapsed + scan.seconds for scan in self.scans(plan.exit, np.inf)}
            if (name.startswith("pair_") or name.startswith("fork_")) and len(times) == 2:
                central[name] = max(times.values())
            elif name.startswith("side_"):
                opposite = "right" if name == "side_left" else "left"
                if opposite in times:
                    crossover[name] = times[opposite]
        if not central or len(crossover) != 2:
            return {"central_max_time": central, "cross_window_time": crossover, "deadline_s": None}
        c, s = min(central.values()), min(crossover.values())
        gap_required = 0.01 + 2 * COMPUTATION_RESERVE_S
        return {
            "central_max_time": central, "cross_window_time": crossover, "gap_s": s - c,
            "deadline_s": (c + s) / 2.0 if s - c > gap_required else None,
        }

    def public_report(self):
        return {
            "direct_native_seconds": {s.name: s.seconds for s in self.initial_scans},
            "programs": {
                n: {
                    "valid": p.valid, "elapsed": p.elapsed, "captures": p.sequence,
                    "terminal_native_seconds": {s.name: s.seconds for s in self.scans(p.exit, np.inf)},
                }
                for n, p in self.programs.items()
            },
            "primary_deadline_s": self.deadline, "public_timing_witness": self.witness,
            "fixed_pair_min_worst_case_time_order": self.fixed_pair,
            "calibration_wall_seconds": self.calibration_seconds,
        }


def best_scan(scans, belief, *, previous_terminal=None):
    if not scans:
        return Choice("no_checked_continuation", (), 0.0)
    scores = [scan_probability(scan.action, belief) - TIME_COST * scan.seconds for scan in scans]
    winner = max(range(len(scans)), key=lambda i: (scores[i], scans[i].name != previous_terminal))
    scan = scans[winner]
    return Choice(scan.name, (scan.name,), float(scores[winner]))


def predicted_event(action, event_id, outcome):
    return PublicObservation(
        event_id, "predicted_patch", action.window, 0.0, "predicted",
        bool(outcome["detected"]), float(outcome["confidence"]), float(outcome["quality"]),
        float("nan"), float("nan"), outcome["candidate_type"],
    )


def value_after_observations(actions, belief, scans, informative, use_observation):
    if not use_observation or not informative or not actions or not scans:
        return best_scan(scans, belief).value
    action, *remaining = actions
    value = 0.0
    outcomes = predict_outcomes(action, belief, geometry_value_=geometry_value(action), informative=True)
    for outcome in outcomes:
        probability = float(outcome["probability"])
        if probability < 1e-7:
            continue
        posterior = deepcopy(belief)
        event_id = f"future_prediction_{len(remaining)}"
        assimilate(posterior, action, predicted_event(action, event_id, outcome), informative)
        value += probability * value_after_observations(remaining, posterior, scans, informative, use_observation)
    return value


def choose_initial(method, model, belief, deadline, informative):
    direct = best_scan(model.scans(model.rig.start, deadline), belief)
    if method in ("A_greedy", "A_geometry"):
        return direct
    if method == "pair_fixed":
        seq = model.fixed_pair
        plan = next(plan for plan in model.programs.values() if plan.sequence == seq)
        available = model.scans(plan.exit, deadline - plan.elapsed)
        if plan.valid and available:
            return Choice("pair_fixed", seq, 0.0)
        return direct
    use_y, use_exit = METHODS[method]
    best = direct
    for name, plan in model.programs.items():
        left = deadline - plan.elapsed
        if not plan.valid or left <= 0:
            continue
        exit_state = plan.exit if use_exit else model.rig.start
        scans = model.scans(exit_state, left)
        if not scans:
            continue
        actions = [action_at(c.capture, c.transfer.output.pose, model.rig.measured) for c in plan.captures]
        value = value_after_observations(actions, belief, scans, informative, use_y) - TIME_COST * plan.elapsed
        if value > best.value + 1e-10:
            best = Choice(name, plan.sequence, float(value))
    return best


def choose_continuation(model, state, belief, remaining, previous_terminal=None):
    scans = model.scans(state, remaining)
    return best_scan(scans, belief, previous_terminal=previous_terminal)
