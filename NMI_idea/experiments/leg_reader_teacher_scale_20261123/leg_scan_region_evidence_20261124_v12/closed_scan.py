"""Legal axial and window programs with a common continuation and persistent evidence."""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

import replay_v11
from evidence import fork_on_tape
from recovery_motion import execute, plan
from recovery_policy import _predict
from recovery_reader import predict as read_v10
from recovery_task import evidence, forecasts, options, table
from study_worlds import truth_at
from track_state import REGIONS, features, from_logits, linear, request
from scipy.special import expit, softmax


KINDS = ("resume", "repeat", "window", "trace_a", "trace_b")
ACTOR_DIM = 76


@dataclass(frozen=True)
class Program:
    region: int
    name: str
    station: int
    sector: int
    anchor: int
    index: int
    visibility: float
    path: object
    key: tuple


@dataclass(frozen=True)
class Read:
    public: dict
    old: dict
    decision: dict
    x: np.ndarray
    hypotheses: dict
    posterior: np.ndarray
    reliability: np.ndarray


class Engine:
    def __init__(self, actor, old, edge=None, decoder=None, blend=0.):
        self.actor, self.old, self.edge, self.decoder, self.blend = actor, old, edge, decoder, blend

    def read(self, live, *, original=False, material=True):
        public = live.scan.prefix.update(live.scan.node)
        old = read_v10(public, self.old, live.case.task)
        if not len(public["frames"]):
            return Read(public, old, old, np.empty((0, 0)), {}, old["role_posterior"], old["target"])
        x, ids = features(public, old, live.scan.env.scanner.surface, self.edge, material=material)
        if self.decoder is not None and not original:
            raw = linear(self.decoder, x)
            post = softmax(raw[:, :8] / self.decoder["temperature"], axis=1)
            reliability = expit(raw[:, 8] / self.decoder["rank_temperature"])
        else:
            post, reliability = old["role_posterior"], old["target"]
        decision = from_logits(public, old, ids, post, reliability, 0. if original else self.blend)
        hypotheses = {r: s for r in REGIONS if (s := request(public, decision, r)) is not None}
        return Read(public, old, decision, x, hypotheses, post, reliability)

    def step(self, live, read=None):
        if read is not None:
            p, d = read.public, read.decision
            live.cache = p, d, evidence(p, d, live)
        choices = options(live)
        x, mask = table(choices)
        logits = _predict(self.actor, x, mask)[0]
        available = np.flatnonzero(mask[1:]) + 1
        select = int(available[np.argmax(logits[available])]) if len(available) and (
            logsumexp(logits[available]) - np.log(len(available)) > logits[0]) else 0
        live.take(*choices[select][:2])
        return live.trace[-1]

    def to_frontier(self, live, region):
        while live.frontier < live.ends[region] and not live.done:
            self.step(live)
        return live

    def finish(self, live, *, update):
        while not live.done:
            read = self.read(live) if update and live.frontier >= live.ends[3] else None
            self.step(live, read)
        at_end = self.read(live, original=not update)
        return at_end, live.scan.grade(at_end.public, at_end.decision)[0]


def _circ(a, b):
    return np.abs((a - b + 12.) % 24. - 12.)


def _anchor_station(surface, station, direction, bounds):
    # Compare the public atlas relationships within a short continuous arc.
    # Change in neighboring named support or target/artery separation marks
    # a section where the two hypotheses predict different geometry.
    tail = station + direction*np.arange(3, 10)
    tail = tail[(tail >= bounds[0]) & (tail <= bounds[1])]
    if not len(tail):
        return None
    atlas = surface.priors[:, tail, :5]
    separation = np.linalg.norm(atlas[:, :, 0] - atlas[:, :, 1], axis=-1)
    available = np.isfinite(separation)
    score = np.sum(np.where(available, separation, 0.), axis=0)/np.maximum(1, available.sum(axis=0))
    if surface.reference_links is not None:
        knots = np.clip(tail, 1, surface.reference_links.shape[1] - 1)
        sides = surface.reference_links[:, knots, :5] != surface.reference_links[:, knots - 1, :5]
        score += 2. * sides.sum(axis=(0, 2))
    return int(tail[np.argmax(score + .14 * np.abs(tail - station))])


def _openings(scan, station, sectors, dwell):
    for a in sectors:
        path = plan(scan, int(station), int(a), dwell=dwell)
        if path is not None:
            return int(a), path
    return None


def programs(live, read, seen=frozenset()):
    scan, result = live.scan, []
    for r, q in read.hypotheses.items():
        if live.frontier < live.ends[r]:
            continue
        f = forecasts(live, q.station)
        visibility = f[:, 0] + .28*f[:, 1] - .08*f[:, 3]
        old_angle = _circ(np.arange(24), q.sector)
        first = int(scan.spec[1])
        # The trace can record bridge observations into an adjacent already
        # visited region. It returns to the lawful continuation pose.
        bounds = first, int(live.frontier)
        for kind in KINDS[1:]:
            key = (r, kind)
            if key in seen:
                continue
            if kind.startswith("trace"):
                role = 0 if kind == "trace_a" else 1
                i = q.indices[role]
                if i < 0:
                    continue
                s = int(np.rint(read.public["stations"][i]))
                a = int(np.rint(read.public["sectors"][i])) % 24
                anchor = _anchor_station(scan.env.scanner.surface, s, -1 if role == 0 else 1, bounds)
                if anchor is None:
                    anchor = _anchor_station(scan.env.scanner.surface, s, 1 if role == 0 else -1, bounds)
                if anchor is None:
                    continue
                order = np.argsort(_circ(np.arange(24), a), kind="stable")[:5]
                chosen = _openings(scan, s, order, 3)
                f0 = forecasts(live, s)
                base = None if chosen is None else f0[chosen[0], 0]
            else:
                i, s, anchor = q.indices[0], q.station, q.station
                order = np.argsort(old_angle, kind="stable")[:4] if kind == "repeat" else np.argsort(
                    -np.where(old_angle >= 2., visibility, -np.inf), kind="stable")[:7]
                chosen = _openings(scan, s, order, 5)
                base = None if chosen is None else visibility[chosen[0]]
            if chosen is not None:
                sector, path = chosen
                result.append(Program(r, kind, s, sector, anchor, i, float(base), path, key))
    return result


def _look(live, a, station, point):
    """Rank a legal next view around the current measured object."""
    surface = live.scan.env.scanner.surface
    sectors = np.flatnonzero(live.scan.env.scanner.valid[station])
    if not len(sectors):
        return sectors
    beam = np.stack([surface.project(station, int(b), point) for b in sectors])
    shadow = np.array([surface.shadow(station, int(b), v) for b, v in zip(sectors, beam, strict=True)])
    f = forecasts(live, station)[sectors]
    value = (.45 * f[:, 0] + .30 * f[:, 1] - .48 * shadow - .020 * np.abs(beam[:, 0])
             - .065*np.maximum(0., 6. - beam[:, 1]) - .050*np.maximum(0., beam[:, 1] - 65.)
             - .013*_circ(sectors, a))
    return sectors[np.argsort(-value, kind="stable")]


def _bridge(live, initial, point, station, start_frame):
    p = live.scan.prefix.update(live.scan.node)
    ids = np.flatnonzero((p["frames"] >= start_frame) & (np.abs(p["stations"] - station) <= 1.35))
    if len(ids):
        distance = np.linalg.norm(p["points"][ids] - point, axis=1)
        i = int(np.argmin(distance + .9*(1. - p["quality"][ids])))
        if distance[i] <= 8.5:
            return p["points"][ids[i]], (int(p["frames"][ids[i]]), int(p["slots"][ids[i]]))
    return initial, None


def execute_program(live, read, task, offset=None):
    branch = live.fork() if offset is None else fork_on_tape(live, offset)
    scan = branch.scan
    origin = (scan.node.station, scan.node.sector)
    start, mm = scan.node.tick, scan.node.distance
    steps = [execute(scan, task.path)]
    pin = read.public["points"][task.index].copy()
    heard = []
    # The first view registers the old candidate. Each bridge then uses its
    # own measured location to select the next legal window.
    while scan.node.station != task.anchor:
        here = int(scan.node.station)
        look = here + int(np.clip(task.anchor - here, -2, 2))
        pin, key = _bridge(branch, pin, pin, here, max(start, steps[-1]["arrival_tick"] - 1))
        if key is not None:
            heard.append(key)
        sectors = _look(branch, scan.node.sector, look, pin)
        admitted = _openings(scan, look, sectors[:6], 2)
        if admitted is None:
            break
        steps.append(execute(scan, admitted[1]))
    reached = (scan.node.station, scan.node.sector)
    if task.name.startswith("trace"):
        _, key = _bridge(branch, pin, pin, reached[0], steps[-1]["arrival_tick"] - 1)
        if key is not None and key not in heard:
            heard.append(key)
    if reached != origin:
        rejoin = plan(scan, *origin, dwell=0)
        if rejoin is None:
            raise RuntimeError("A checked evidence trace could not return to its scan pose")
        steps.append(execute(scan, rejoin))
    branch.cache = None
    if scan.node.tick != len(scan.node.reader.frames) or (scan.node.station, scan.node.sector) != origin:
        raise RuntimeError("Contact trace and sensor clock disagree")
    detail = {"region": task.region+1, "program": task.name, "keys": read.hypotheses[task.region].keys,
              "target": [task.station, task.sector], "anchor_station": task.anchor, "reached": list(reached),
              "start_tick": int(start), "end_tick": int(scan.node.tick),
              "bridge_keys": heard, "travel_mm": float(scan.node.distance - mm),
              "visited": [[s["next_station"], s["next_sector"], s["next_tick"]] for s in steps]}
    return branch, detail


def snapshot(live, read, scope):
    p = read.public
    truth = truth_at(live.scan.node, p, live.scan.tape)
    return {"public": p, **truth, "layout_id": live.scan.tape.layout.identifier, "scope": scope,
            "route": live.case.task, "station": int(live.scan.node.station), "tick": int(live.scan.node.tick),
            "trajectory_id": live.case.identifier, "complete": live.scan.node.station == live.scan.spec[2]}


def action_vector(live, read, task):
    code = np.zeros(len(KINDS))
    code[KINDS.index(task.name if task else "resume")] = 1.
    row = np.zeros(24) if task is None else read.hypotheses[task.region].state
    scanner = live.scan
    continuation = np.zeros(6)
    late = []
    for r in range(6):
        i = np.flatnonzero(read.public["regions"] == r)
        if len(i):
            d = read.decision
            a = i[np.argmax(d["winner_score"][i])]
            continuation[r] = d["score"][a] >= 0.
            late.extend((float(d["target"][a]), float(d["role_posterior"][a, 1]), d["winner_score"][a]))
        else:
            late.extend((0., 0., -1.))
    if task is None:
        action = np.zeros(16)
    else:
        f = forecasts(live, task.station)[task.sector]
        action = np.r_[task.region/5., (task.station - scanner.spec[1])/160., (task.anchor - task.station)/10.,
                       task.sector/24., task.visibility, task.path.ticks/100., f,
                       _circ(scanner.node.sector, task.sector)/12.,
                       float(task.region == 4), (live.frontier - live.ends[task.region])/50.,
                       (task.station - scanner.node.station)/100.]
    point = np.r_[code, row, continuation, late, action, live.frontier/160., scanner.node.station/160.,
                  float(scanner.node.station == scanner.spec[2]), len(live.trace)/24.,
                  np.log1p(len(read.public["frames"]))/8., np.sin(scanner.node.sector*np.pi/12.),
                  np.cos(scanner.node.sector*np.pi/12.)]
    if point.shape != (ACTOR_DIM,) or not np.isfinite(point).all():
        raise RuntimeError(f"Invalid region-program input {point.shape}")
    return point.astype(np.float32)
