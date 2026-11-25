"""Identity-aware public acquisition on the existing checked contact chart."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
from scipy.special import ndtr

V13 = Path(__file__).resolve().parents[1] / "leg_scan_student_history_20261124_v13"
sys.path.insert(1, str(V13))

import trajectory_v13 as tr
from closed_scan import ACTOR_DIM, KINDS, Program, _anchor_station, _circ, _openings, action_vector
from evidence import fork_on_tape
from forward_anchor import step_slot
from recovery_motion import execute as move, plan
from recovery_task import forecasts, motion
from region_policy import VALUE_WEIGHTS
from track_state import REGIONS, Request, linear

import identity_v15


SECTORS = np.arange(24)
ANGLES = SECTORS * np.pi / 12.
NORMALS = np.column_stack((np.cos(ANGLES), np.sin(ANGLES)))
TANGENTS = np.column_stack((-NORMALS[:, 1], NORMALS[:, 0]))


def _key(public, index):
    return int(public["frames"][index]), int(public["slots"][index])


def _lookup(public, key):
    frames = public["frames"]
    start = int(np.searchsorted(frames, key[0]))
    end = int(np.searchsorted(frames, key[0], side="right"))
    return next((i for i in range(start, end) if public["slots"][i] == key[1]), None)


def _trusted(read, q):
    return q is not None and q.indices[0] >= 0 and read.decision["score"][q.indices[0]] >= 0.


class _Windows:
    def __init__(self, live, read):
        self.live = live
        self.read = read
        self.scan = live.scan
        self.surface = self.scan.env.scanner.surface
        self.base = {}
        self.values = {}
        self.source_weights = self.scan.node.reader.w.reshape(len(self.surface.priors), -1).sum(axis=1)

    def _base(self, station):
        if station not in self.base:
            self.base[station] = forecasts(self.live, station)
        return self.base[station]

    def renew_motion(self):
        # Same identity belief throughout the macro. The causal population
        # forecast can still incorporate each newly acquired physical frame.
        self.base.clear()
        self.values.clear()
        self.source_weights = self.scan.node.reader.w.reshape(len(self.surface.priors), -1).sum(axis=1)

    def _projection(self, station, index):
        public, decision = self.read.public, self.read.decision
        at = int(np.rint(public["stations"][index]))
        observed = public["points"][index]
        if at == station:
            return observed
        source = self.surface.priors
        roles = decision["role_posterior"][index, :5]
        choices = source[:, station, :5] - source[:, at, :5]
        linked = np.isfinite(choices).all(axis=-1)
        if self.surface.reference_links is not None:
            start, end = sorted((at, station))
            linked &= self.surface.reference_links[:, start:end, :5].all(axis=1)
        weights = self.source_weights[:, None] * roles[None, :] * linked
        if weights.sum() <= 1.e-9:
            return observed + self.surface.cross_origin[station] - self.surface.cross_origin[at]
        return observed + np.einsum("sr,srd->d", weights, np.nan_to_num(choices)) / weights.sum()

    def _geometry(self, station, points):
        # At the discrete contact poses, ray lookup is Surface.shadow's exact
        # public chart calculation. Targets are observations or atlas points.
        drop = points[None, :, :] - self.surface.contact[station, :, None, :]
        lateral = np.einsum("apd,ad->ap", drop, TANGENTS)
        depth = -np.einsum("apd,ad->ap", drop, NORMALS)
        u = np.clip(np.rint((lateral + 25.) / 2.5), 0, len(self.surface.offsets) - 1).astype(int)
        z = np.clip(np.rint((depth - 4.) / 2.), 0, len(self.surface.depths) - 1).astype(int)
        shadow = self.surface.ray[station, SECTORS[:, None], u, z]
        aperture = (ndtr((25. - lateral) / 2.35) - ndtr((-25. - lateral) / 2.35))
        aperture *= ndtr((80. - depth) / 1.85) - ndtr((3. - depth) / 1.85)
        detected = .74 * aperture * (1. - .92 * shadow)
        return detected, aperture, shadow

    def near(self, station):
        region = motion.region_of(self.scan.spec, station)
        q = self.read.hypotheses.get(region)
        if q is not None:
            return q
        others = [s for s in self.read.hypotheses.values() if _trusted(self.read, s)]
        return min(others, key=lambda s: abs(s.station - station), default=None)

    def features(self, station, q=None, index=None):
        """Marginalize public candidate and atlas windows with the same belief."""
        r = motion.region_of(self.scan.spec, station) if q is None else q.region
        if index is None and _trusted(self.read, q):
            index = q.indices[0]
        if index is not None and index < 0:
            raise ValueError("An observed window requires a real echo address")
        rival = q.indices[1] if q is not None and index == q.indices[0] and _trusted(self.read, q) else -1
        key = station, r, index, rival
        if key in self.values:
            return self.values[key]
        single = self._base(station)
        source = self.surface.priors[:, station, 0]
        present = np.isfinite(source).all(axis=1)
        if not np.any(present) and index is None:
            self.values[key] = single
            return single
        points = list(source[present])
        atlas = self.source_weights[present]
        atlas = atlas / atlas.sum() if atlas.sum() > 0. else np.full(len(atlas), 1. / max(1, len(atlas)))
        unknown = self.read.decision.get("unknown_probability", np.zeros(6))[r]
        if index is None:
            weight = atlas
        else:
            points.append(self._projection(station, index))
            distance = abs(float(self.read.public["stations"][index]) - station)
            mass = float(np.clip(1. - unknown, 0., 1.) * np.exp(-distance / 24.)) if len(atlas) else 1.
            if rival >= 0:
                points.append(self._projection(station, rival))
                prob = self.read.decision.get("candidate_probability", self.read.decision["target"])[[index, rival]]
                rivals = prob / max(1.e-9, float(prob.sum())) if prob.sum() > 0. else np.array([1., 0.])
            else:
                rivals = np.array([1.])
            weight = np.r_[atlas * (1. - mass), mass * rivals]
        detect, aperture, shadow = self._geometry(station, np.asarray(points))
        result = single.copy()
        # Keep the population uncertainty/separation input of the frozen actor.
        # Detection, acoustic shadow and aperture now also score the measured
        # object hypotheses, including an explicit atlas term for unknown.
        blend = .60 + .15 * float(unknown)
        result[:, 0] = (1. - blend) * single[:, 0] + blend * (detect @ weight)
        result[:, 2] = (1. - blend) * single[:, 2] + blend * ((detect * detect) @ weight)
        result[:, 3] = (1. - blend) * single[:, 3] + blend * (shadow @ weight)
        result[:, 4] = (1. - blend) * single[:, 4] + blend * (aperture @ weight)
        result[~self.surface.valid[station]] = 0.
        self.values[key] = result
        return result


def _merit(features):
    return features[:, 0] + .28 * features[:, 1] - .08 * features[:, 3]


def _missing_request(live, read, region):
    scan = live.scan
    first, last = int(live.ends[region - 1] + 1), int(live.ends[region])
    middle = (first + last) // 2
    available = np.flatnonzero(scan.env.scanner.valid[first:last + 1].any(axis=1)) + first
    if not len(available):
        return None
    station = int(available[np.argmin(np.abs(available - middle))])
    valid = np.flatnonzero(scan.env.scanner.valid[station])
    sector = int(valid[np.argmax(_merit(_Windows(live, read).features(station))[valid])])
    risk = 1.22
    state = np.zeros(24, dtype=np.float32)
    state[20], state[-1] = -1., risk
    return Request(region, (-1, -1), (None, None), tuple(np.empty(0, int) for _ in range(2)),
                   station, sector, risk, state)


def _review(live, read, seen):
    windows = _Windows(live, read)
    scan, result = live.scan, []
    for r, q in read.hypotheses.items():
        if live.frontier < live.ends[r]:
            continue
        trusted = _trusted(read, q)
        reference = q if trusted or q.indices[0] < 0 else _missing_request(live, read, r)
        if reference is None:
            continue
        bounds = int(scan.spec[1]), int(live.frontier)
        for name in KINDS[1:]:
            key = r, name
            if key in seen:
                continue
            order = None
            if name.startswith("trace"):
                alternate = name == "trace_b"
                index = q.indices[int(alternate)]
                if not trusted or index < 0:
                    continue
                s = int(np.rint(read.public["stations"][index]))
                direction = 1 if alternate else -1
                anchor = _anchor_station(windows.surface, s, direction, bounds)
                if anchor is None:
                    anchor = _anchor_station(windows.surface, s, -direction, bounds)
                if anchor is None:
                    continue
                value = _merit(windows.features(s, q, index))
                dwell = 3
            else:
                index = q.indices[0] if trusted else -1
                s = anchor = reference.station
                value = _merit(windows.features(s, q))
                arc = _circ(SECTORS, reference.sector)
                order = arc <= 1.5 if name == "repeat" else arc >= 2.
                dwell = 5
            legal = np.flatnonzero(windows.surface.valid[s] if order is None else windows.surface.valid[s] & order)
            order = legal[np.argsort(-value[legal], kind="stable")]
            chosen = _openings(scan, s, order, dwell)
            if chosen is not None:
                sector, path = chosen
                result.append(Program(r, name, s, sector, anchor, index, float(value[sector]), path, key))
    return result


def _bridge_echo(before, previous_index, after, station, point, start_frame):
    public, decision = after.public, after.decision
    frames = public["frames"]
    rows = np.flatnonzero((frames >= start_frame) & (np.abs(public["stations"] - station) <= 1.35))
    if not len(rows):
        return None
    old = _lookup(public, _key(before.public, previous_index))
    old_roles = before.decision["role_posterior"][previous_index]
    identities = decision.get("object_id", decision["track_id"])
    same = identities[rows] == identities[old] if old is not None else np.zeros(len(rows), bool)
    overlap = np.sqrt(decision["role_posterior"][rows] * old_roles).sum(axis=1)
    quality = public["quality"][rows]
    dist = np.linalg.norm(public["points"][rows] - point, axis=1)
    rank = np.where(dist <= 8.5, dist + .9 * (1. - quality) - 2.0 * same - 2.0 * overlap, np.inf)
    if np.isfinite(rank).any():
        return int(rows[np.argmin(rank)])
    return None


def _advance(live, read, goal, windows, refresh):
    scan, node = live.scan, live.scan.node
    start, angle, tick, mm = int(node.station), int(node.sector), int(node.tick), float(node.distance)
    route = scan.env.route
    views = []
    belief_tick = tick
    region = motion.region_of(scan.spec, start)
    while node.station < goal.station:
        station = int(node.station) + 1
        sectors = route.sectors[int(node.sector)]
        legal = route.bounded_candidates(node, scan.spec, scan.limits)
        q = windows.near(station)
        f = windows.features(station, q)
        rank = _merit(f)[sectors]
        if station < scan.spec[2]:
            following = _merit(windows.features(station + 1, windows.near(station + 1)))
            lookahead = np.max(np.where(route.legal[station, sectors], following[route.sectors[sectors]], -1.), axis=1)
            rank = rank + .43 * lookahead
        if not legal.any():
            raise RuntimeError("No legal continuation for the public belief")
        slot = int(np.argmax(np.where(legal, rank, -np.inf)))
        chosen = int(sectors[slot])
        _, physical = step_slot(scan.env, scan.spec, node, scan.tape, scan.limits, slot)
        view_region = motion.region_of(scan.spec, station)
        known = _trusted(read, q)
        unknown = read.decision.get("unknown_probability", np.zeros(6))[q.region if q is not None else view_region]
        views.append({"station": station, "sector": chosen, "region": view_region + 1,
                      "visibility": float(f[chosen, 0]), "belief_score": float(rank[slot]),
                      "belief_tick": belief_tick, "key": q.keys[0] if known else None,
                      "unknown": float(unknown),
                      "frozen_mpc_sector": int(sectors[np.argmax(physical["candidate_scores"])])})
        if view_region != region and station < goal.station:
            read = refresh(live)
            windows = _Windows(live, read)
            belief_tick = int(node.tick)
            region = view_region
        else:
            windows.renew_motion()
    live.frontier = max(live.frontier, node.station)
    live.cache = None
    if len(node.reader.frames) != node.tick or tick >= node.tick:
        raise RuntimeError("Motion did not advance the physical sample clock")
    detail = {"skill": "advance", "goal_region": goal.region + 1, "goal_station": goal.station,
              "goal_sector": -1, "dwell": 0, "sweep": 0., "station": start, "sector": angle, "tick": tick,
              "next_station": int(node.station), "next_sector": int(node.sector), "next_tick": int(node.tick),
              "ticks": int(node.tick - tick), "distance_mm": float(node.distance - mm), "frontier": int(live.frontier),
              "earlier_region": False, "dird": False, "executed_sweep": None, "belief_windows": views}
    live.trace += (detail,)
    return detail


def vectors(live, read, choices):
    return np.stack([action_vector(live, read, task) for task in choices])


def chooser(actor, rule):
    def choose(live, read, choices):
        if not choices:
            return None
        score = linear(actor, vectors(live, read, [None, *choices])) @ VALUE_WEIGHTS
        advantage = score[1:] - score[0]
        advantage += [rule.get("r5_weight", 0.) * (p.region == 4) + rule.get("r6_bias", 0.) * (p.region == 5)
                      for p in choices]
        best = int(np.argmax(advantage))
        return choices[best] if advantage[best] > rule["margin"] else None
    return choose


class Controller:
    vectors = staticmethod(vectors)
    chooser = staticmethod(chooser)

    def __init__(self, reader, legacy_reader, legacy_mode="new"):
        self.reader = reader
        self.legacy_reader = legacy_reader
        self.legacy_mode = legacy_mode

    def read(self, live):
        read = identity_v15.apply(tr.ENGINE.read(live), self.reader, self.legacy_reader, self.legacy_mode)
        hypotheses = dict(read.hypotheses)
        for r in REGIONS:
            if r not in hypotheses and live.frontier >= live.ends[r]:
                fallback = _missing_request(live, read, r)
                if fallback is not None:
                    hypotheses[r] = fallback
        return replace(read, hypotheses=hypotheses)

    def menu(self, live, read, seen=frozenset()):
        return _review(live, read, seen)

    def execute(self, live, read, task, offset=None):
        branch = live.fork() if offset is None else fork_on_tape(live, offset)
        scan = branch.scan
        origin = (scan.node.station, scan.node.sector)
        start, mm = int(scan.node.tick), float(scan.node.distance)
        steps = [move(scan, task.path)]
        picked, heard = [], []
        current, index = read, task.index
        if task.name.startswith("trace"):
            if index < 0:
                raise RuntimeError("A bridge requires a measured starting object")
            q = read.hypotheses[task.region]
            while True:
                here = int(scan.node.station)
                previous = _Windows(branch, current)._projection(here, index)
                updated = self.read(branch)
                match = _bridge_echo(current, index, updated, here, previous, max(start, steps[-1]["arrival_tick"] - 1))
                if match is not None:
                    key = _key(updated.public, match)
                    if key not in heard:
                        heard.append(key)
                    current, index = updated, match
                else:
                    key = _key(current.public, index)
                    at = _lookup(updated.public, key)
                    if at is None:
                        raise RuntimeError("A bridge lost its observed prefix address")
                    current, index = updated, at
                if here == task.anchor:
                    break
                look = here + int(np.clip(task.anchor - here, -2, 2))
                windows = _Windows(branch, current)
                f = windows.features(look, current.hypotheses.get(task.region, q), index)
                value = _merit(f)
                sectors = np.flatnonzero(windows.surface.valid[look])
                order = sectors[np.argsort(-value[sectors], kind="stable")]
                admitted = _openings(scan, look, order, 2)
                if admitted is None:
                    break
                sector, path = admitted
                picked.append({"station": int(look), "sector": sector, "source_key": key,
                               "belief_tick": int(scan.node.tick), "visibility": float(value[sector])})
                steps.append(move(scan, path))
        reached = (scan.node.station, scan.node.sector)
        if reached != origin:
            rejoin = plan(scan, *origin, dwell=0)
            if rejoin is None:
                raise RuntimeError("A checked evidence trace could not return to its scan pose")
            steps.append(move(scan, rejoin))
        branch.cache = None
        if scan.node.tick != len(scan.node.reader.frames) or (scan.node.station, scan.node.sector) != origin:
            raise RuntimeError("Contact trace and sensor clock disagree")
        detail = {"region": task.region + 1, "program": task.name, "keys": read.hypotheses[task.region].keys,
                  "target": [task.station, task.sector], "anchor_station": task.anchor, "reached": list(reached),
                  "start_tick": start, "end_tick": int(scan.node.tick), "bridge_keys": heard, "bridge_windows": picked,
                  "target_key": _key(read.public, task.index) if task.index >= 0 else None,
                  "atlas_fallback": task.index < 0, "visibility": task.visibility,
                  "travel_mm": float(scan.node.distance - mm),
                  "visited": [[s["next_station"], s["next_sector"], s["next_tick"]] for s in steps]}
        return branch, detail

    def resume(self, live, read):
        if live.done:
            return None
        if live.frontier < live.ends[2]:
            return tr.ENGINE.step(live)
        goal = live.base_goal()
        if goal.skill == "submit":
            return live.take(goal)
        read = self.read(live) if read is None else read
        return _advance(live, read, goal, _Windows(live, read), self.read)

    def roll(self, live, chooser, *, seen=frozenset(), stop=None, first_read=None, before_action=None, after_action=None):
        used, current, actions = set(seen), first_read, []
        while not live.done:
            if live.frontier < live.ends[2]:
                self.resume(live, None)
                current = None
                continue
            if current is None:
                current = self.read(live)
            choices = self.menu(live, current, used)
            if stop is not None and stop(live, current, choices, used):
                return tr.Result(live, current, actions, frozenset(used), prefix=True)
            task = chooser(live, current, choices)
            if task is None:
                self.resume(live, current)
                if not live.done:
                    current = None
                continue
            previous = current
            captured = None if before_action is None else before_action(live, previous, task)
            used.add(task.key)
            live, detail = self.execute(live, previous, task)
            current = self.read(live)
            detail["before_last"] = bool(live.frontier < live.scan.spec[2])
            if after_action is not None:
                after_action(live, previous, current, task, captured, detail)
            actions.append(detail)
        if current is None:
            current = self.read(live)
        return tr.Result(live, current, actions, frozenset(used))

    @staticmethod
    def grade(result):
        return tr.score(result)
