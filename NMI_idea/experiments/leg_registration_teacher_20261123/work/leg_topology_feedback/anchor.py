"""Trace a registered vascular anchor and return at the common scan speed."""
from dataclasses import dataclass

import numpy as np
import motion

from teacher_benchmark import route_path


@dataclass(frozen=True)
class AnchorPlan:
    anchor_station: int
    anchor_sector: int
    kind: str
    alternate_return: bool
    poses: np.ndarray
    lengths: np.ndarray
    entry: tuple

    @property
    def ticks(self):
        return len(self.lengths)

    @property
    def distance(self):
        return float(self.lengths.sum())


def find_bridge(manifest, spec, stations):
    source = manifest["sources"]["Open3DModel right"]
    bridge = next(e for e in source["bridge_edges"] if "Posterior_tibial" in e["right_name"])
    start_mm = bridge["left_station_mm"]
    station = int(np.clip(np.searchsorted(-stations, start_mm, side="right") - 1, spec[1], spec[2]))
    return station


def ambiguous(history, probabilities, spec, station, threshold, trigger):
    """Evidence from at least two distinct material stations supports a decision."""
    region = motion.region_of(spec, station)
    confidence_by_station = {}
    for frame, probability in zip(reversed(history.samples), reversed(probabilities)):
        if trigger == "local" and frame.station < station - 5.:
            break
        if frame.station > station or frame.region != region or not probability.shape[1]:
            continue
        s = int(np.floor(frame.station + .5))
        v = float(probability[0].max())
        confidence_by_station[s] = max(v, confidence_by_station.get(s, 0.))
    highs = sum(v >= threshold for v in confidence_by_station.values())
    return highs < 2


def stable_stations(history, station):
    by_station = {}
    for frame, prob in zip(history.samples, history.online):
        s = int(np.floor(frame.station + .5))
        if s >= station - 3 or not prob.shape[1]:
            continue
        confidence = float(prob[0].max())
        if confidence >= .91:
            by_station[s] = max(confidence, by_station.get(s, 0.))
    # A recent corroborating observation is useful as a local track anchor.
    # The anatomical landmark itself is evaluated first.
    return [s for s in sorted(by_station, reverse=True)
            if s - 1 in by_station or s + 1 in by_station]


def edge_poses(env, spec, start, sectors):
    path = [[float(start), float(sectors[0])]]
    for i, (sector, following) in enumerate(zip(sectors, sectors[1:])):
        choices = np.flatnonzero(env.route.sectors[sector] == following)
        if len(choices) != 1:
            raise RuntimeError("An anchor branch left the common scan action grid")
        segment, _ = route_path(env, spec, start + i, sector, int(choices[0]))
        path.extend(segment)
    poses = np.asarray(path, dtype=float)
    for i in range(1, len(poses)):
        # Unwrap each edge using the shortest angular step for contact validation.
        poses[i, 1] = poses[i-1, 1] + motion._sector_delta(poses[i-1, 1], poses[i, 1])
    return poses


def choose_return(env, forecast, first, last, entry_sector, return_sector, travel_weight):
    """Choose a legal second view while preserving both endpoint anchors."""
    route = env.route
    value = np.full(motion.SECTORS, -np.inf)
    value[return_sector] = 0.
    choices = np.empty((last - first, motion.SECTORS), dtype=np.int8)
    for s in range(last - 1, first - 1, -1):
        observation = np.array([forecast(s + 1, a)[1] if env.scanner.valid[s + 1, a] else 0.
                                for a in range(motion.SECTORS)])
        target = route.sectors
        scores = observation[target] - travel_weight * route.distance[s] + value[target]
        scores = np.where(route.legal[s], scores, -np.inf)
        choices[s - first] = np.argmax(scores, axis=1)
        value = np.max(scores, axis=1)
    if not np.isfinite(value[entry_sector]):
        return None
    sectors = [entry_sector]
    for s in range(last - first):
        sectors.append(int(route.sectors[sectors[-1], choices[s, sectors[-1]]]))
    if sectors[-1] != return_sector:
        raise RuntimeError("Planning did not return to the active anchor")
    return sectors


def reserve(node, spec, limits):
    i, a = int(node.station) - spec[1], int(node.sector)
    ticks = limits.ticks - node.tick - int(limits.reserve_ticks[i, a])
    distance = limits.mm - node.distance - float(limits.reserve_mm[i, a])
    return ticks, distance


def build_plan(env, spec, node, limits, anchors, forecast, trail, *, allow_local=True):
    current, sector = int(node.station), int(node.sector)
    ticks, distance = reserve(node, spec, limits)
    for first, kind in anchors:
        if kind == "track" and not allow_local:
            continue
        if first not in trail or current - first < 3 or current - first > ticks // 2:
            continue
        # Even a perfectly straight return still consumes twice the axial distance.
        if 2. * abs(env.scanner.surface.y[current] - env.scanner.surface.y[first]) > distance + 1e-7:
            continue
        historic = [trail[s] for s in range(first, current + 1)]
        outbound = edge_poses(env, spec, first, historic)[::-1]
        paths = []
        for weight in (.013, .031):
            other = choose_return(env, forecast, first, current, historic[0], sector, weight)
            if other is not None and other != historic and other not in paths:
                paths.append(other)
        paths.append(historic)
        for return_sectors in paths:
            inbound = edge_poses(env, spec, first, return_sectors)
            inbound[:, 1] += outbound[-1, 1] - inbound[0, 1]
            # One held frame at the turn and one at the return anchor.
            poses = np.concatenate((outbound, inbound, inbound[-1:]), axis=0)
            if len(poses) - 1 > ticks:
                continue
            lengths = env.scanner._path_metrics(poses, spec)
            if lengths is None or lengths.sum() > distance + 1e-7:
                continue
            return AnchorPlan(first, historic[0], kind, return_sectors != historic,
                              poses, lengths, (current, sector))
    return None


def execute(env, spec, node, trial, limits, plan, *, repeat=False, mute=False):
    if mute and not repeat:
        raise ValueError("The timing control waits at the entry anchor")
    if (int(node.station), int(node.sector)) != plan.entry:
        raise ValueError("Recheck started at a different route anchor")
    ticks, distance = reserve(node, spec, limits)
    if plan.ticks > ticks or (not repeat and plan.distance > distance + 1e-7):
        raise RuntimeError("Recheck would consume the reserved continuation budget")
    if repeat:
        positions = np.repeat(np.asarray(plan.entry, float)[None], plan.ticks, axis=0)
        lengths = np.zeros(plan.ticks)
    else:
        positions, lengths = plan.poses[1:], plan.lengths
    before = node.tick
    for (s, a), length in zip(positions, lengths):
        node.tick += 1
        node.distance += float(length)
        if not mute:
            env.scanner._observe(node, spec, trial, float(s), float(a), node.tick)
    last = positions[-1] if mute else node.reader.frames[-1]
    if (abs(last[0] - node.station) > 1e-7 or abs(motion._sector_delta(node.sector, last[1])) > 1e-7
            or node.tick - before != plan.ticks):
        raise RuntimeError("Recheck failed to return to the same physical route anchor")
    return node


def find_prefix(env, spec, node, trial, limits, history, forecast, bridge, threshold, *, allow_local, trigger):
    """Check predeclared encounter stations online, with no future observations."""
    trail = {}
    examined, no_plan, resolved = 0, 0, 0
    station_checks = ((980, 1010, 1040, 1080, 1120, 1160, 1200, 1230) if trigger == "local"
                      else (1035, 1075, 1115, 1155, 1195, 1230))
    while True:
        history.append(node.reader.frames)
        current = int(node.station)
        trail[current] = int(node.sector)
        if int(-env.scanner.stations[current]) in station_checks:
            if ambiguous(history, history.online, spec, current, threshold, trigger):
                examined += 1
                if ambiguous(history, history.smoothed(), spec, current, threshold, trigger):
                    candidates = [(bridge, "junction")] + [(s, "track") for s in stable_stations(history, current)]
                    plan = build_plan(env, spec, node, limits, candidates, forecast, trail, allow_local=allow_local)
                    if plan is not None:
                        return node, plan, {"ambiguous_checks": examined, "unreachable_checks": no_plan,
                                            "history_resolved_checks": resolved}
                    no_plan += 1
                else:
                    resolved += 1
        if current == spec[2]:
            return node, None, {"ambiguous_checks": examined, "unreachable_checks": no_plan,
                                "history_resolved_checks": resolved}
        env.route.advance(node, spec, trial, limits)
