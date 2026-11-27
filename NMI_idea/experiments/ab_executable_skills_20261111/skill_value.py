"""Exact value of executable sensing programs in the frozen acoustic graph."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "ab_soft_exploration_20261111"
sys.path.insert(0, str(REFERENCE))
from program_value import accepted, decision_prior, observation_tables, update
from run_soft_stress import route, scan_at
from toy_world import Action, Fixture, legal_actions


ARMS = (
    "A_task",
    "A_fixed_goal",
    "B_actual",
    "B_nominal",
    "B_no_lookahead",
    "B_MAP",
    "A_legacy",
)
SIDES = ("L", "R")
TOL = 1.e-12
RESPONSE_MAPS = np.array(list(product((0, 1), repeat=5)), dtype=int)
FIXED_MAPS = RESPONSE_MAPS[[0, -1]]


@dataclass(frozen=True)
class Statistics:
    correct: float
    wrong: float
    frames: float

    @property
    def value(self) -> float:
        return self.correct / self.frames if self.frames else 0.

    @property
    def unresolved(self) -> float:
        return max(0., 1. - self.correct - self.wrong)

    def as_dict(self) -> dict[str, float]:
        return dict(correct=self.correct, wrong=self.wrong, frames=self.frames,
                    unresolved=self.unresolved, correct_per_frame=self.value)


@dataclass(frozen=True)
class Skill:
    """The scheduled prefix and the terminal goal for each first packet."""

    path: tuple[Action, ...]
    goal: str | None
    goals_by_packet: tuple[str, ...]
    statistics: Statistics
    nominal_exit: bool = False

    def as_dict(self) -> dict:
        return dict(path=[action.name for action in self.path], goal=self.goal,
                    goals_by_packet=self.goals_by_packet, nominal_exit=self.nominal_exit,
                    **self.statistics.as_dict())


@lru_cache(maxsize=None)
def terminal_path(node: str, side: str, fixture: Fixture,
                  already_captured_at_terminal: bool = False) -> tuple[Action, ...]:
    terminal = "T" + side
    approach = route(node, terminal, fixture)
    if approach is None:
        raise ValueError(f"no legal route from {node} to {terminal}")
    scan = scan_at(terminal, fixture)
    if already_captured_at_terminal:
        # An acquisition already taken by the prefix still counts in the
        # nominal-exit ablation. Arrival on a hypothetical approach can only
        # be the second terminal acquisition.
        return approach or (scan,)
    if approach:
        return (*approach, scan)
    return (scan, scan)


def _gate_from_prediction(predicted: np.ndarray, emission: np.ndarray,
                          target: np.ndarray, action: Action) -> np.ndarray:
    if action.dest not in ("TL", "TR"):
        return np.zeros(len(predicted), dtype=bool)
    public_probability = predicted @ emission[:, 4]
    target_probability = np.divide(
        predicted @ target[:, 4], public_probability,
        out=np.zeros(len(predicted), dtype=float), where=public_probability > 0.)
    return target_probability >= .90


def path_statistics(path: tuple[Action, ...], fixture: Fixture,
                    scoring_mass: np.ndarray, public_mass: np.ndarray) -> np.ndarray:
    """Return unnormalized (correct, wrong, frames) masses.

    The scoring mass may condition on one visibility mode. At every simulated
    packet the same full public posterior is used for acceptance.
    """
    scoring = np.asarray(scoring_mass, dtype=float)[None, :]
    public = np.asarray(public_mass, dtype=float)[None, :]
    result = np.zeros(3, dtype=float)
    for action in path:
        result[2] += scoring.sum()
        if not len(scoring):
            break
        transition, emission, target, _, _ = observation_tables(action, fixture)
        sp = scoring @ transition
        pp = public @ transition
        stops = _gate_from_prediction(pp, emission, target, action)
        if stops.any():
            result[0] += (sp[stops] @ target[:, 4]).sum()
            result[1] += (sp[stops] @ (emission[:, 4] - target[:, 4])).sum()
        scoring = sp[:, None, :] * emission.T[None, :, :]
        public = pp[:, None, :] * emission.T[None, :, :]
        keep = scoring.sum(axis=-1) > 0.
        keep[stops, 4] = False
        scoring = scoring[keep]
        public = public[keep]
    return result


def _stats(values: np.ndarray) -> Statistics:
    return Statistics(*map(float, values))


def _retain_maximum(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    maximum = np.max(values[indices])
    return indices[values[indices] >= maximum - TOL]


def _best_map(table: np.ndarray, prefix: np.ndarray, maps: np.ndarray,
              pending: str) -> tuple[tuple[str, ...], Statistics]:
    # Optimizing the aggregate ratio requires the joint response map.
    totals = table[np.arange(5)[None, :], maps].sum(axis=1) + prefix
    ids = np.arange(len(maps))
    ids = _retain_maximum(totals[:, 0] / totals[:, 2], ids)
    ids = _retain_maximum(-totals[:, 1], ids)
    ids = _retain_maximum(-totals[:, 2], ids)
    ids = _retain_maximum(np.sum(maps == SIDES.index(pending), axis=1), ids)
    i = ids[0]
    return tuple(SIDES[k] for k in maps[i]), _stats(totals[i])


def _feedback_skill(first: Action, fixture: Fixture, belief: np.ndarray,
                    scoring: np.ndarray, maps: np.ndarray, pending: str,
                    nominal_exit: bool) -> Skill:
    transition, emission, target, _, _ = observation_tables(first, fixture)
    sp, pp = scoring @ transition, belief @ transition
    stops = bool(_gate_from_prediction(pp[None, :], emission, target, first)[0])
    prefix = np.array([
        sp @ target[:, 4] if stops else 0.,
        sp @ (emission[:, 4] - target[:, 4]) if stops else 0.,
        1.,
    ])
    table = np.zeros((5, 2, 3), dtype=float)
    for packet in range(5):
        if packet == 4 and stops:
            continue
        scoring_mass = sp * emission[:, packet]
        public_mass = pp * emission[:, packet]
        if scoring_mass.sum() <= 0.:
            continue
        start = "C" if nominal_exit else first.dest
        for k, side in enumerate(SIDES):
            path = terminal_path(start, side, fixture,
                                 already_captured_at_terminal=first.dest == "T" + side)
            table[packet, k] = path_statistics(path, fixture, scoring_mass, public_mass)
    goals, statistics = _best_map(table, prefix, maps, pending)
    return Skill((first,), None, goals, statistics, nominal_exit)


def _choose_best(candidates: list[Skill], pending: str) -> Skill:
    # All methods use the same tie rules and exact score.
    maximum = max(s.statistics.value for s in candidates)
    choices = [s for s in candidates if s.statistics.value >= maximum - TOL]
    minimum = min(s.statistics.wrong for s in choices)
    choices = [s for s in choices if s.statistics.wrong <= minimum + TOL]
    minimum = min(s.statistics.frames for s in choices)
    choices = [s for s in choices if s.statistics.frames <= minimum + TOL]
    return min(choices, key=lambda s: (
        -(s.goals_by_packet.count(pending) if s.goals_by_packet else 5 * (s.goal == pending)),
        s.path[0].side != pending,
        tuple(a.name for a in s.path), s.goals_by_packet, s.goal or "",
    ))


@lru_cache(maxsize=16_384)
def plan(arm: str, node: str, fixture: Fixture, belief_key: tuple[float, ...],
         pending: str) -> Skill:
    """Select one complete executable program from the shared action library."""
    belief = np.asarray(belief_key, dtype=float)
    scoring = decision_prior(belief, "hard" if arm == "B_MAP" else "soft")
    candidates = []
    for side in SIDES:
        path = terminal_path(node, side, fixture)
        candidates.append(Skill(path, side, (), _stats(path_statistics(path, fixture, scoring, belief))))
    if arm in ("A_task", "B_no_lookahead"):
        return _choose_best(candidates, pending)
    if arm not in ("A_fixed_goal", "B_actual", "B_nominal", "B_MAP"):
        raise ValueError(f"unknown planning arm {arm}")
    for first in legal_actions(node, fixture):
        candidates.append(_feedback_skill(
            first, fixture, belief, scoring,
            FIXED_MAPS if arm == "A_fixed_goal" else RESPONSE_MAPS, pending,
            nominal_exit=arm == "B_nominal"))
    return _choose_best(candidates, pending)


def actual_statistics(skill: Skill, fixture: Fixture, belief: np.ndarray) -> Statistics:
    """Re-evaluate a saved program using its real predicted exit and full belief."""
    if not skill.goals_by_packet:
        return _stats(path_statistics(skill.path, fixture, belief, belief))
    first = skill.path[0]
    transition, emission, target, _, _ = observation_tables(first, fixture)
    predicted = belief @ transition
    stops = bool(_gate_from_prediction(predicted[None, :], emission, target, first)[0])
    result = np.array([
        predicted @ target[:, 4] if stops else 0.,
        predicted @ (emission[:, 4] - target[:, 4]) if stops else 0.,
        1.,
    ])
    for packet, side in enumerate(skill.goals_by_packet):
        if packet == 4 and stops:
            continue
        mass = predicted * emission[:, packet]
        path = terminal_path(first.dest, side, fixture,
                             already_captured_at_terminal=first.dest == "T" + side)
        result += path_statistics(path, fixture, mass, mass)
    return _stats(result)
