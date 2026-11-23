"""Contact-sampled material recovery and bounded local sweeps for the v9 Scan."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from operator import index
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

V9 = Path(__file__).resolve().parents[1] / "leg_scan_full_route_20261124_v9"
if str(V9) not in sys.path:
    sys.path.insert(1, str(V9))

from scan_v9 import held_frames, motion
from environment import ACTION_MM, ACTION_TICKS
from forward_anchor import _path_for_slot


@dataclass(frozen=True)
class _Segment:
    positions: np.ndarray
    costs: np.ndarray


@dataclass(frozen=True)
class Plan:
    """A pose-specific recovery plan. Clock cost is exact, distance is nominal."""

    station: int
    sector: int
    dwell: int
    sweep: float
    use_dird: bool
    travel_ticks: int
    travel_mm: float
    sweep_ticks: int
    sweep_mm: float
    reserve_ticks: int
    reserve_mm: float
    dird_metadata: dict
    _route: object = field(repr=False, compare=False)
    _bounds: tuple = field(repr=False)
    _origin: tuple = field(repr=False)
    _segments: tuple = field(repr=False, compare=False)
    _probe: object = field(repr=False, compare=False)
    _sweep_cap_mm: float = field(repr=False)

    @property
    def ticks(self):
        return self.travel_ticks + self.dwell + self.sweep_ticks

    @property
    def distance_mm(self):
        return self.travel_mm + self.sweep_mm

    @property
    def budget_distance_mm(self):
        """Upper bound on executed distance, allowing checked feedback updates."""
        return self.travel_mm + self._sweep_cap_mm


class _ContactGraph:
    """Public chart graph with common axial slots and adjacent-sector rotations."""

    def __init__(self, env, first, last):
        self.env, self.first, self.last = env, first, last
        self.size = (last - first + 1) * motion.SECTORS
        route = env.route
        axial_station, axial_sector, slot = np.nonzero(route.legal[first:last])
        axial_station += first
        next_sector = route.sectors[axial_sector, slot]
        lower = self.vertex(axial_station, axial_sector)
        upper = self.vertex(axial_station + 1, next_sector)
        axial_ticks = route.clock[axial_station, axial_sector, slot]
        axial_mm = route.distance[axial_station, axial_sector, slot]

        station, sector = np.indices((last - first + 1, motion.SECTORS))
        station += first
        fraction = np.linspace(0., 1., 21)
        sample_sector = sector[..., None] + fraction
        sample_station = np.broadcast_to(station[..., None], sample_sector.shape)
        world, legal = env.scanner.surface.path_coordinates(sample_station, sample_sector)
        arcs = np.linalg.norm(np.diff(world, axis=-2), axis=-1)
        good = legal.all(axis=-1)
        turns = self.vertex(station[good], sector[good])
        next_turns = self.vertex(station[good], (sector[good] + 1) % motion.SECTORS)
        turn_ticks = np.maximum(1, np.ceil(1.035 * arcs[good].max(axis=-1) * 20 /
                                         (motion.CLOCK_S * motion.MAX_SPEED_MM_S))).astype(int)
        turn_mm = arcs[good].sum(axis=-1) + .015

        self._from = np.r_[lower, turns]
        self._to = np.r_[upper, next_turns]
        self._ticks = np.r_[axial_ticks, turn_ticks]
        self._mm = np.r_[axial_mm, turn_mm]
        self._slots = np.r_[slot, np.full(len(turns), -1)]
        self._edge_ids = {(int(start), int(end)): (i, False)
                          for i, (start, end) in enumerate(zip(self._from, self._to, strict=True))}
        self._edge_ids.update({(end, start): (i, True) for (start, end), (i, _) in tuple(self._edge_ids.items())})
        self._sampled = {}
        self._enabled = np.ones(len(self._slots), dtype=bool)
        self._graphs = {}
        self._version = 0

    def vertex(self, station, sector):
        return (station - self.first) * motion.SECTORS + sector

    def pose(self, vertex):
        row, sector = divmod(vertex, motion.SECTORS)
        return self.first + row, sector

    @lru_cache(maxsize=192)
    def _tree(self, source, order, version):
        if order not in self._graphs:
            enabled = self._enabled
            starts, ends = self._from[enabled], self._to[enabled]
            ticks, mm = self._ticks[enabled], self._mm[enabled]
            rank = (ticks + .06 * mm, ticks + 1.e-6 * mm, mm + 1.e-6 * ticks)[order]
            self._graphs[order] = csr_matrix((np.tile(rank, 2), (np.r_[starts, ends], np.r_[ends, starts])),
                                             shape=(self.size, self.size))
        return dijkstra(self._graphs[order], indices=source, return_predecessors=True)[1]

    def _segment(self, start, end, spec):
        edge, reverse = self._edge_ids[start, end]
        if (edge, reverse) not in self._sampled:
            station, sector = self.pose(int(self._from[edge]))
            slot = int(self._slots[edge])
            if slot >= 0:
                try:
                    positions, costs = _path_for_slot(
                        self.env, spec, SimpleNamespace(station=station, sector=sector), slot,
                    )
                except RuntimeError:
                    return self._disable(edge)
            else:
                ticks = int(self._ticks[edge])
                chart = np.column_stack((np.full(ticks + 1, station),
                                         sector + np.arange(ticks + 1, dtype=float) / ticks))
                costs = self.env.scanner._path_metrics(chart, spec)
                if costs is None:
                    return self._disable(edge)
                positions = chart[1:]
            backward = np.vstack((positions[:-1][::-1], [station, sector]))
            self._sampled[edge, False] = _Segment(positions, costs)
            self._sampled[edge, True] = _Segment(backward, costs[::-1])
        return self._sampled[edge, reverse]

    def _disable(self, edge):
        self._enabled[edge] = False
        self._graphs.clear()
        self._version += 1
        return None

    def path(self, origin, target, spec, order):
        start, end = self.vertex(*origin), self.vertex(*target)
        while True:
            tree = self._tree(start, order, self._version) if start != end else None
            cursor = end
            path = []
            while cursor != start:
                previous = int(tree[cursor])
                if previous < 0:
                    return None
                segment = self._segment(previous, cursor, spec)
                if segment is None:
                    break
                path.append(segment)
                cursor = previous
            if cursor == start:
                return tuple(reversed(path))


@lru_cache(maxsize=24)
def _graph(env, first, last):
    return _ContactGraph(env, first, last)


def _pose(node):
    if int(node.station) != node.station or int(node.sector) != node.sector:
        raise ValueError("Recovery starts and ends at discrete chart poses")
    return int(node.station), int(node.sector)


def _clock_limit(scan):
    # Trial.branch keeps the original total horizon and exposes the suffix on
    # its own clock. The last usable absolute index is the same in both cases.
    return min(scan.limits.ticks, len(scan.tape.tape.normal) - 1)


def _reserve(scan, station, sector):
    first, last = scan.spec[1:3]
    if not first <= station <= last or not 0 <= sector < motion.SECTORS:
        return None
    if not scan.env.scanner.valid[station, sector]:
        return None
    ticks = int(scan.limits.reserve_ticks[station - first, sector])
    distance = float(scan.limits.reserve_mm[station - first, sector])
    return (ticks, distance) if np.isfinite(distance) else None


def can_finish(scan):
    """Check the common legal MPC reserve at the current material pose."""
    reserve = _reserve(scan, *_pose(scan.node))
    return bool(reserve is not None and scan.node.tick + reserve[0] <= _clock_limit(scan)
                and scan.node.distance + reserve[1] <= scan.limits.mm + 1.e-7)


def _plan_sweep(scan, station, sector, sweep, clock, distance, use_dird):
    # Zero axial command keeps the entire loop at the reached station.
    material = scan.env.material
    proxy = SimpleNamespace(station=station, sector=sector)
    controls = np.array([sweep, 0., .45 * sweep], dtype=float)
    probe = material.plan_probe(proxy, scan.spec, 1, controls, min(clock, ACTION_TICKS),
                                min(distance, ACTION_MM), use_dird=use_dird)
    if probe is None:
        return None
    cost = sum(float(material._plan_costs(item[0], scan.spec, item[4]).sum()) for item in probe.segments)
    cap = min(distance, ACTION_MM, cost + max(.5, .25 * cost))
    return probe, cost, cap


def plan(scan, station: int, sector: int, dwell: int = 3, sweep: float = 0.0, *, use_dird=False) -> Plan | None:
    """Admit exact-pose recovery if contact and terminal clock/distance permit."""
    station, sector, dwell = index(station), index(sector), index(dwell)
    sweep = float(sweep)
    if dwell < 0 or not np.isfinite(sweep) or not -1. <= sweep <= 1.:
        raise ValueError("Dwell must be nonnegative and sweep must lie in [-1, 1]")
    if scan.done:
        return None
    origin = _pose(scan.node)
    target = station, sector
    reserve = _reserve(scan, *target)
    if reserve is None or not scan.env.scanner.valid[origin]:
        return None
    start_tick, start_mm = int(scan.node.tick), float(scan.node.distance)
    clock = _clock_limit(scan) - start_tick - reserve[0] - dwell
    distance = scan.limits.mm - start_mm - reserve[1]
    if clock < 0 or distance < -1.e-7:
        return None
    distance = max(0., distance)
    if abs(origin[0] - station) > clock:
        return None
    surface = scan.env.scanner.surface
    chord = np.linalg.norm(surface.world(*origin) - surface.world(*target))
    if chord > min(distance, clock * motion.CLOCK_S * motion.MAX_SPEED_MM_S) + motion.EPS:
        return None

    graph = _graph(scan.env, *scan.spec[1:3]) if origin != target else None
    # The common weighted route preference is tried first. The two resource
    # extremes can still admit a plan when one cap binds before the other.
    for order in range(3) if graph is not None else (0,):
        segments = graph.path(origin, target, scan.spec, order) if graph is not None else ()
        if segments is None:
            return None
        travel_ticks = sum(len(segment.costs) for segment in segments)
        travel_mm = sum((float(segment.costs.sum()) for segment in segments), 0.)
        if travel_ticks > clock or travel_mm > distance + motion.EPS:
            continue
        if sweep:
            shaped = _plan_sweep(scan, station, sector, sweep, clock - travel_ticks,
                                 max(0., distance - travel_mm), use_dird)
            if shaped is None:
                continue
            probe, sweep_mm, sweep_cap = shaped
            dird = dict(probe.dird_metadata)
            dird["requested"] = bool(use_dird)
            dird["used"] = bool(dird["chart_score_applied"])
            dird["physical_execution"] = False
            if use_dird and not dird["available"]:
                dird["selection_status"] = "dird_unavailable_policy_shape"
            elif use_dird and not dird["used"]:
                dird["selection_status"] = "dird_unsupported_chart_pose_policy_shape"
        elif travel_ticks or dwell:
            probe, sweep_mm, sweep_cap = None, 0., 0.
            dird = {"requested": bool(use_dird), "available": bool(scan.env.material.dird.metadata["available"]),
                    "used": False, "chart_score_applied": False, "physical_execution": False,
                    "selection_status": "no_sweep"}
        else:
            return None
        return Plan(station, sector, dwell, sweep, bool(use_dird), travel_ticks, travel_mm,
                    probe.ticks if probe is not None else 0, sweep_mm, *reserve, dird,
                    scan.env.route, tuple(scan.spec[1:3]), (*origin, start_tick, start_mm),
                    segments, probe, sweep_cap)
    return None


def execute(scan, plan: Plan) -> dict:
    """Acquire a recovery on this Scan's tape and leave it at the achieved pose."""
    if not isinstance(plan, Plan):
        raise TypeError("Recovery requires an admitted Plan")
    origin = (*_pose(scan.node), int(scan.node.tick), float(scan.node.distance))
    if scan.done or plan._route is not scan.env.route or plan._bounds != tuple(scan.spec[1:3]) or origin != plan._origin:
        raise ValueError("Recovery plan does not match this scan pose, clock and chart")
    station, sector, start_tick, start_mm = origin
    reserve = _reserve(scan, plan.station, plan.sector)
    if (reserve is None or start_tick + plan.ticks + reserve[0] > _clock_limit(scan)
            or start_mm + plan.budget_distance_mm + reserve[1] > scan.limits.mm + motion.EPS):
        raise ValueError("Recovery plan no longer fits the scan budget")
    node = scan.node.fork()
    scanner = scan.env.scanner
    trajectory = []

    for segment in plan._segments:
        for (s, a), cost in zip(segment.positions, segment.costs, strict=True):
            node.tick += 1
            node.distance += float(cost)
            node.station, node.sector = float(s), float(motion.wrap_sector(a))
            scanner._observe(node, scan.spec, scan.tape, node.station, node.sector, node.tick)
            trajectory.append([int(node.tick), node.station, node.sector, float(cost), node.distance - start_mm])
    if _pose(node) != (plan.station, plan.sector):
        raise RuntimeError("Sampled recovery missed the requested material pose")
    # Integer typing is required by the common forward MPC; the sampled pose
    # is already exactly this station and cyclic sector.
    node.station, node.sector = _pose(node)
    arrival_tick = int(node.tick)
    dwell_mm = node.distance - start_mm
    held_frames(scan.env, scan.spec, node, scan.tape, plan.dwell)
    trajectory.extend([tick, node.station, node.sector, 0., dwell_mm]
                      for tick in range(arrival_tick + 1, node.tick + 1))

    probe = None
    if plan._probe is not None:
        result = scan.env.material.run_probe(node, scan.spec, scan.tape, plan._probe, plan.sweep_ticks,
                                            plan._sweep_cap_mm, feedback=True, use_dird=plan.use_dird,
                                            copy_node=False)
        if not result.legal or _pose(result.node) != (plan.station, plan.sector):
            raise RuntimeError(f"Reserved recovery sweep failed its contact check: {result.reason}")
        node = result.node
        probe_path = result.trajectory.copy()
        probe_path[:, 4] += dwell_mm
        trajectory.extend(probe_path.tolist())
        probe = {"goal": plan._probe.goal.tolist(), "fraction": result.applied_fraction,
                 "feedback_updates": result.feedback_updates, "ticks": result.ticks,
                 "distance_mm": result.distance, "shape": plan._probe.shape.tolist()}
    if (node.tick - start_tick != plan.ticks or len(node.reader.frames) != node.tick
            or node.tick + reserve[0] > _clock_limit(scan)
            or node.distance + reserve[1] > scan.limits.mm + motion.EPS):
        raise RuntimeError("Recovery violated frame or terminal reserve bookkeeping")
    scan.node = node
    return {"station": station, "sector": sector, "tick": start_tick,
            "next_station": node.station, "next_sector": node.sector, "next_tick": int(node.tick),
            "arrival_tick": arrival_tick, "ticks": plan.ticks, "duration": plan.ticks,
            "distance_mm": node.distance - start_mm, "travel_ticks": plan.travel_ticks,
            "travel_mm": plan.travel_mm, "dwell": plan.dwell, "sweep": probe, "dird": plan.dird_metadata,
            "use_dird": plan.dird_metadata["used"],
            "reserve_ticks": reserve[0], "reserve_mm": reserve[1], "trajectory": trajectory}
