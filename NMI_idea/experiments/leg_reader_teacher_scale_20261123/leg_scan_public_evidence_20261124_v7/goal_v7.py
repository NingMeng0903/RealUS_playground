"""Public, chart-legal planning of a short evidence goal.

This module is intentionally independent of a trial, a scanner layout, or a
label.  It plans only on the public route chart supplied by the caller.  A
goal is a short sequence of the same one-station MPC slots used by v1--v6,
with an optional public terminal opportunity attached to its final station
and acoustic window.  The caller may then execute the returned slots and
replan after new observations.

The route object follows the v5 convention:

* ``route.sectors`` is normally ``[sector, slot]`` and contains the target
  sector for each one-station slot;
* ``route.legal`` is normally ``[station, sector]``; and
* ``route.clock`` and ``route.distance`` are ``[station, sector, slot]``.

For the v7 runner a fully expanded ``[station, sector, slot]`` form is also
accepted for ``sectors`` and ``legal``.  No hidden anatomy, simulator object,
or actual ``Trial`` is imported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

import numpy as np


_EPS = 1.0e-9
_MIN_SPAN = 3
_MAX_SPAN = 8


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from a mapping or a small limits object."""

    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def normalise_valid(base: Any, legal: Any) -> np.ndarray:
    """Min--max normalise only finite legal scores.

    Illegal entries, NaNs, and infinities are excluded from the extrema and
    returned as ``-inf``.  Consequently padding a score vector with arbitrary
    illegal/non-finite entries cannot change any valid result.  If all valid
    values are equal, valid entries are zero rather than NaN.
    """

    values = np.asarray(base, dtype=float).reshape(-1)
    mask = np.asarray(legal, dtype=bool).reshape(-1)
    if values.shape != mask.shape:
        raise ValueError(f"base and legal must have the same shape, got {values.shape} and {mask.shape}")
    valid = mask & np.isfinite(values)
    out = np.full(values.shape, -np.inf, dtype=float)
    if not np.any(valid):
        return out
    finite = values[valid]
    low = float(np.min(finite))
    high = float(np.max(finite))
    if high - low <= _EPS:
        out[valid] = 0.0
    else:
        out[valid] = (finite - low) / (high - low)
    return out


@dataclass(frozen=True)
class Goal:
    """A public evidence target and its exact chart-legal MPC prefix."""

    station: int
    sector: int
    slots: tuple[int, ...]
    steps: tuple[Mapping[str, Any], ...]
    score: float
    object_ids: tuple[Any, ...] = ()
    relation: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly public record of this planned goal."""

        return {
            "station": int(self.station),
            "sector": int(self.sector),
            "slots": [int(slot) for slot in self.slots],
            "steps": [_plain(step) for step in self.steps],
            "score": float(self.score),
            "object_ids": [_plain(value) for value in self.object_ids],
            "relation": str(self.relation),
        }


def _plain(value: Any) -> Any:
    """Convert numpy scalars/containers used in audit records to plain data."""

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sector_slots(route: Any, station: int, sector: int) -> np.ndarray:
    values = np.asarray(route.sectors)
    if values.ndim == 3:
        result = values[int(station), int(sector)]
    elif values.ndim == 2:
        result = values[int(sector)]
    elif values.ndim == 1:
        result = values
    else:
        raise ValueError(f"route.sectors must be 1D, 2D, or 3D, got {values.shape}")
    result = np.asarray(result, dtype=int).reshape(-1)
    if not len(result):
        raise ValueError("route.sectors has no MPC slots")
    return result


def _slot_vector(route: Any, name: str, station: int, sector: int,
                 slot_count: int) -> np.ndarray:
    values = np.asarray(getattr(route, name))
    if values.ndim == 3:
        result = values[int(station), int(sector)]
    elif values.ndim == 2:
        # The compact form is useful for a toy route and means one value per
        # station/sector.  A [station, slot] or [sector, slot] array is also
        # accepted when its second axis identifies the MPC slots.
        if values.shape[0] > int(station) and values.shape[1] == slot_count:
            result = values[int(station)]
        elif values.shape[0] > int(station) and values.shape[1] > int(sector):
            result = np.repeat(values[int(station), int(sector)], slot_count)
        elif values.shape[0] > int(sector) and values.shape[1] == slot_count:
            result = values[int(sector)]
        else:
            raise ValueError(f"route.{name} shape {values.shape} cannot index {(station, sector)}")
    elif values.ndim == 1:
        if len(values) != slot_count:
            raise ValueError(f"route.{name} has {len(values)} slots, expected {slot_count}")
        result = values
    else:
        raise ValueError(f"route.{name} must be 1D, 2D, or 3D, got {values.shape}")
    result = np.asarray(result, dtype=float).reshape(-1)
    if len(result) != slot_count:
        raise ValueError(f"route.{name} returned {len(result)} slots, expected {slot_count}")
    return result


def _legal_slots(route: Any, station: int, sector: int, slot_count: int) -> np.ndarray:
    values = np.asarray(route.legal)
    if values.ndim == 3:
        result = values[int(station), int(sector)]
    elif values.ndim == 2:
        # v5 stores one legality bit per source state; expanded v7 routes may
        # use [station, sector] only as well.  A row with slot_count entries is
        # treated as an explicit slot mask for compact test routes.
        if values.shape[0] > int(station) and values.shape[1] > int(sector):
            scalar = values[int(station), int(sector)]
            result = np.repeat(bool(scalar), slot_count)
        elif values.shape[0] > int(station) and values.shape[1] == slot_count:
            result = values[int(station)]
        else:
            raise ValueError(f"route.legal shape {values.shape} cannot index {(station, sector)}")
    elif values.ndim == 1:
        if len(values) != slot_count:
            raise ValueError(f"route.legal has {len(values)} slots, expected {slot_count}")
        result = values
    else:
        raise ValueError(f"route.legal must be 1D, 2D, or 3D, got {values.shape}")
    result = np.asarray(result, dtype=bool).reshape(-1)
    if len(result) != slot_count:
        raise ValueError(f"route.legal returned {len(result)} slots, expected {slot_count}")
    return result


def _limits_value(limits: Any, name: str, default: float) -> float:
    value = _field(limits, name, default)
    if value is None:
        return float(default)
    value = float(value)
    if value < 0. or not np.isfinite(value):
        if np.isinf(value) and value > 0.:
            return value
        raise ValueError(f"limits.{name} must be a non-negative finite value or +inf")
    return value


def _reserve_table(route: Any, limits: Any, bounds: tuple[int, int], name: str) -> np.ndarray:
    """Get the common v5 reserve table, deriving it from ``route.reserve``."""

    value = _field(limits, name, None)
    if value is not None:
        table = np.asarray(value, dtype=float)
    else:
        reserve = getattr(route, "reserve", None)
        if reserve is None:
            raise ValueError(f"limits.{name} is required when route.reserve is unavailable")
        result = reserve(int(bounds[0]), int(bounds[1]))
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            raise ValueError("route.reserve must return (ticks, distance)")
        table = np.asarray(result[0 if name == "reserve_ticks" else 1], dtype=float)
    if table.ndim not in (1, 2):
        raise ValueError(f"limits.{name} must be one- or two-dimensional, got {table.shape}")
    return table


def _reserve_at(table: np.ndarray, station: int, sector: int,
                bounds: tuple[int, int]) -> float:
    start, end = (int(bounds[0]), int(bounds[1]))
    row_offset = int(station) - start
    # v5 limits are indexed by station offset.  Accept a full-route table too.
    if table.ndim == 1:
        row = row_offset if 0 <= row_offset < len(table) else int(station)
        if not 0 <= row < len(table):
            return float("inf")
        value = table[row]
    else:
        if table.shape[0] == end - start + 1:
            row = row_offset
        elif 0 <= int(station) < table.shape[0]:
            row = int(station)
        else:
            row = row_offset
        if not 0 <= row < table.shape[0] or not table.shape[1]:
            return float("inf")
        col = int(sector) % table.shape[1]
        value = table[row, col]
    value = float(value)
    return value if np.isfinite(value) and value >= 0. else float("inf")


@dataclass(frozen=True)
class _Path:
    station: int
    sector: int
    tick: int
    distance: float
    slots: tuple[int, ...]
    steps: tuple[Mapping[str, Any], ...]


def _pareto(paths: list[_Path]) -> list[_Path]:
    """Keep exact non-dominated clock/distance alternatives per endpoint."""

    if len(paths) <= 1:
        return paths
    ordered = sorted(paths, key=lambda item: (item.tick, item.distance, item.slots))
    kept: list[_Path] = []
    best_distance = float("inf")
    for item in ordered:
        if item.distance < best_distance - _EPS:
            kept.append(item)
            best_distance = item.distance
    return kept


def _terminal_value(terminal_value: Any, station: int, sector: int) -> float:
    if terminal_value is None:
        return 0.0
    if callable(terminal_value):
        value = terminal_value(int(station), int(sector))
    elif isinstance(terminal_value, Mapping):
        value = terminal_value.get((int(station), int(sector)), 0.0)
    else:
        raise TypeError("terminal_value must be a mapping, callable, or None")
    value = float(value)
    return value if np.isfinite(value) else float("-inf")


def _step(path: _Path, slot: int, target: int, ticks: float, distance: float) -> dict[str, Any]:
    before_tick = int(path.tick)
    after_tick = before_tick + int(ticks)
    before_distance = float(path.distance)
    after_distance = before_distance + float(distance)
    return {
        "from_station": int(path.station),
        "from_sector": int(path.sector),
        "station": int(path.station + 1),
        "sector": int(target),
        "slot": int(slot),
        "ticks": int(ticks),
        "tick_before": before_tick,
        "tick_after": after_tick,
        "distance_mm": float(distance),
        "distance_before_mm": before_distance,
        "distance_after_mm": after_distance,
        "state": {"station": int(path.station + 1), "sector": int(target)},
        "time": int(after_tick),
    }


def choose_goal(route: Any, limits: Any, bounds: tuple[int, int], start_sector: int,
                tick: int, distance: float, terminal_value: Any = None, *,
                object_ids: Any = (), relation: str = "") -> Goal | None:
    """Choose a bounded public goal with an exact legal MPC-slot prefix.

    The search considers destination spans from three through eight stations.
    Every edge is checked against ``route.legal``, finite non-negative clock and
    distance, the global envelope, and the common ``reserve_ticks/mm`` needed
    to continue from the proposed endpoint to ``bounds[1]``.  Pareto clock /
    distance paths are retained per endpoint, so a cheap path is not allowed
    to hide a slower but shorter alternative.  Missing terminal opportunities
    have value zero; non-finite explicitly supplied values are rejected.
    """

    if len(bounds) != 2:
        raise ValueError("bounds must be a (start, end) pair")
    start, end = int(bounds[0]), int(bounds[1])
    if end <= start:
        raise ValueError(f"bounds must increase, got {bounds}")
    if end - start < _MIN_SPAN:
        return None
    current_tick = int(tick)
    current_distance = float(distance)
    if current_tick < 0 or current_distance < 0. or not np.isfinite(current_distance):
        raise ValueError("tick and distance must be non-negative and finite")
    max_tick = _limits_value(limits, "ticks", float("inf"))
    max_distance = _limits_value(limits, "mm", float("inf"))
    reserve_ticks = _reserve_table(route, limits, (start, end), "reserve_ticks")
    reserve_mm = _reserve_table(route, limits, (start, end), "reserve_mm")

    start_slots = _sector_slots(route, start, int(start_sector))
    nslots = len(start_slots)
    # Check the start sector early, because silent modulo wrapping at this
    # boundary can hide a caller's chart-coordinate error.
    if int(start_sector) < 0:
        raise ValueError("start_sector must be non-negative")
    states: dict[int, list[_Path]] = {
        int(start_sector): [_Path(start, int(start_sector), current_tick,
                                  current_distance, (), ())]
    }
    goals: list[tuple[float, int, float, float, _Path]] = []
    max_station = min(end, start + _MAX_SPAN)

    for station in range(start + 1, max_station + 1):
        next_states: dict[int, list[_Path]] = {}
        previous = states
        for source_sector, paths in previous.items():
            sectors = _sector_slots(route, station - 1, source_sector)
            if len(sectors) != nslots:
                raise ValueError("route.sectors has an inconsistent slot count")
            legal = _legal_slots(route, station - 1, source_sector, nslots)
            clocks = _slot_vector(route, "clock", station - 1, source_sector, nslots)
            lengths = _slot_vector(route, "distance", station - 1, source_sector, nslots)
            for path in paths:
                for slot, target in enumerate(sectors):
                    if not bool(legal[slot]):
                        continue
                    step_clock = clocks[slot]
                    step_distance = lengths[slot]
                    if (not np.isfinite(step_clock) or not np.isfinite(step_distance)
                            or step_clock < 1. or step_distance < 0.):
                        continue
                    new_tick = path.tick + int(step_clock)
                    new_distance = path.distance + float(step_distance)
                    target = int(target)
                    if new_tick > max_tick or new_distance > max_distance + _EPS:
                        continue
                    remaining_tick = _reserve_at(reserve_ticks, station, target, (start, end))
                    remaining_distance = _reserve_at(reserve_mm, station, target, (start, end))
                    if (not np.isfinite(remaining_tick) or not np.isfinite(remaining_distance)
                            or new_tick + remaining_tick > max_tick
                            or new_distance + remaining_distance > max_distance + _EPS):
                        continue
                    step = _step(path, slot, target, step_clock, step_distance)
                    candidate = _Path(station, target, new_tick, new_distance,
                                      path.slots + (int(slot),),
                                      path.steps + (step,))
                    next_states.setdefault(target, []).append(candidate)
        states = {sector: _pareto(paths) for sector, paths in next_states.items()}
        if station - start >= _MIN_SPAN:
            for sector, paths in states.items():
                value = _terminal_value(terminal_value, station, sector)
                if not np.isfinite(value):
                    continue
                for path in paths:
                    # Terminal value is primary.  Small deterministic costs
                    # break ties without changing an actual public opportunity.
                    score = value - 1.0e-3 * float(path.tick - current_tick)
                    score -= 1.0e-5 * float(path.distance - current_distance)
                    goals.append((score, station, path.distance, float(path.tick), path))
        if not states:
            break

    if not goals:
        return None
    # Highest public value wins; ties prefer an earlier goal and then the
    # shorter exact path.  No hidden label is used as a tie breaker.
    _, _, _, _, selected = max(
        goals,
        key=lambda item: (item[0], -item[1], -item[2], -item[3], tuple(-v for v in item[4].slots)),
    )
    goal = Goal(station=selected.station, sector=selected.sector,
                slots=selected.slots, steps=selected.steps,
                score=float(next(item[0] for item in goals if item[4] is selected)),
                object_ids=tuple(object_ids), relation=str(relation))
    validate_goal(route, limits, (start, end), start_sector, tick, distance, goal)
    return goal


def validate_goal(route: Any, limits: Any, bounds: tuple[int, int], start_sector: int,
                  tick: int, distance: float, goal: Goal) -> None:
    """Directly validate a returned goal against the public chart and envelope."""

    start, end = int(bounds[0]), int(bounds[1])
    if not start <= int(goal.station) <= end:
        raise ValueError("goal station is outside bounds")
    if len(goal.slots) != int(goal.station) - start:
        raise ValueError("goal slots do not cover the claimed station span")
    reserve_ticks = _reserve_table(route, limits, (start, end), "reserve_ticks")
    reserve_mm = _reserve_table(route, limits, (start, end), "reserve_mm")
    max_tick = _limits_value(limits, "ticks", float("inf"))
    max_distance = _limits_value(limits, "mm", float("inf"))
    state_station, state_sector = start, int(start_sector)
    state_tick, state_distance = int(tick), float(distance)
    if state_sector < 0:
        raise ValueError("start_sector must be non-negative")
    if len(goal.steps) != len(goal.slots):
        raise ValueError("goal steps and slots have different lengths")
    for index, slot in enumerate(goal.slots):
        if state_station >= end:
            raise ValueError("goal path runs beyond terminal station")
        sectors = _sector_slots(route, state_station, state_sector)
        nslots = len(sectors)
        legal = _legal_slots(route, state_station, state_sector, nslots)
        clocks = _slot_vector(route, "clock", state_station, state_sector, nslots)
        lengths = _slot_vector(route, "distance", state_station, state_sector, nslots)
        slot = int(slot)
        if not 0 <= slot < nslots or not legal[slot]:
            raise ValueError(f"goal uses illegal slot {slot} at {(state_station, state_sector)}")
        step_clock, step_distance = float(clocks[slot]), float(lengths[slot])
        if not np.isfinite(step_clock) or step_clock < 1. or not np.isfinite(step_distance) or step_distance < 0.:
            raise ValueError("goal uses a non-finite or invalid route edge")
        target = int(sectors[slot])
        state_tick += int(step_clock)
        state_distance += step_distance
        if state_tick > max_tick or state_distance > max_distance + _EPS:
            raise ValueError("goal exceeds the common motion envelope")
        reserve_t = _reserve_at(reserve_ticks, state_station + 1, target, (start, end))
        reserve_d = _reserve_at(reserve_mm, state_station + 1, target, (start, end))
        if state_tick + reserve_t > max_tick or state_distance + reserve_d > max_distance + _EPS:
            raise ValueError("goal leaves no common legal suffix")
        state_station, state_sector = state_station + 1, target
        public_step = goal.steps[index]
        if int(public_step.get("station", state_station)) != state_station or int(public_step.get("sector", state_sector)) != state_sector:
            raise ValueError("goal public step state does not match the route")
    if state_station != int(goal.station) or state_sector != int(goal.sector):
        raise ValueError("goal endpoint does not match its slot path")


def self_test() -> dict[str, Any]:
    """Run small CPU-only chart and padding checks used by the v7 handoff."""

    class ToyRoute:
        sectors = np.empty((6, 4, 3), dtype=int)
        legal = np.ones((6, 4, 3), dtype=bool)
        clock = np.ones((6, 4, 3), dtype=float)
        distance = np.ones((6, 4, 3), dtype=float)

        def reserve(self, first, last):
            n = int(last) - int(first) + 1
            return np.zeros((n, 4)), np.zeros((n, 4))

    route = ToyRoute()
    for station in range(6):
        for sector in range(4):
            route.sectors[station, sector] = (np.arange(3) + sector) % 4
    # One illegal edge demonstrates that a chart-invalid branch is not used.
    route.legal[1, 0, 1] = False
    limits = {"ticks": 20, "mm": 20., "reserve_ticks": np.zeros((7, 4)),
              "reserve_mm": np.zeros((7, 4))}
    goal = choose_goal(route, limits, (0, 5), 0, 0, 0., {(3, 3): 2.0, (4, 0): .1},
                       object_ids=("public-obj-1",), relation="same_object")
    if goal is None or goal.station != 3 or goal.sector != 3:
        raise AssertionError(f"toy goal selection failed: {goal}")
    validate_goal(route, limits, (0, 5), 0, 0, 0., goal)
    first = normalise_valid([1., 3., 2., np.nan], [True, True, False, True])
    padded = normalise_valid([1., 3., 2., np.nan, 10_000., -np.inf],
                             [True, True, False, True, False, False])
    if not np.allclose(first[:3], padded[:3]) or not np.isneginf(padded[3]):
        raise AssertionError("normalise_valid is not padding invariant")
    return {"goal": goal.as_dict(), "padding_invariant": True}


__all__ = ["Goal", "choose_goal", "normalise_valid", "validate_goal", "self_test"]


if __name__ == "__main__":
    import json

    print(json.dumps(self_test(), indent=2, sort_keys=True))
