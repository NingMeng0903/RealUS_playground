"""Public identity hypotheses and equal-time physical ultrasound revisits."""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
V10 = ROOT.parent / "leg_scan_recovery_20261124_v10"
sys.path.insert(1, str(V10))

from recovery_task import forecasts
from recovery_motion import execute, plan
from recovery_reader import predict
from study_worlds import truth_at


@dataclass(frozen=True)
class Anchor:
    key: tuple
    station: float
    sector: float
    point: np.ndarray
    embedding: np.ndarray
    old_views: frozenset


@dataclass(frozen=True)
class IdentityQuery:
    region: int
    anchors: tuple
    features: np.ndarray
    incumbent_score: float
    incumbent_target: float
    incumbent_accepted: bool


@dataclass(frozen=True)
class View:
    name: str
    station: int
    sector: int
    dwell: int
    ticks: int
    features: np.ndarray
    visibility: float


def _keys(public):
    return list(zip(public["frames"].tolist(), public["slots"].tolist(), strict=True))


def _view_keys(public):
    station = np.floor(public["stations"] + .5).astype(int)
    sector = np.floor(public["sectors"] + .5).astype(int) % 24
    return station, sector


def _anchor(public, prediction, index, keys):
    ds = np.abs(public["stations"] - public["stations"][index])
    distance = np.linalg.norm(public["points"] - public["points"][index], axis=1)
    stable = (ds <= 2.25) & (distance <= 4.0)
    a, b = _view_keys(public)
    return Anchor(keys[index], float(public["stations"][index]), float(public["sectors"][index]),
                  public["points"][index].copy(), prediction["embedding"][index].copy(),
                  frozenset(zip(a[stable].tolist(), b[stable].tolist(), strict=True)))


def query(public, prediction, region):
    """Freeze old image addresses and a spatially different alternative."""
    indices = np.flatnonzero(public["regions"] == region)
    if not len(indices):
        return None
    score = prediction["winner_score"]
    target = prediction["target"]
    a = int(indices[np.argmax(score[indices])])
    station = np.abs(public["stations"][indices] - public["stations"][a])
    distance = np.linalg.norm(public["points"][indices] - public["points"][a], axis=1)
    nearby = (station <= 4.25) & (distance >= 3.8) & (distance <= 22.)
    contenders = indices[nearby]
    b = None
    if len(contenders):
        rank = (.56 * target[contenders] + .34 * (1. - prediction["null"][contenders])
                + .10 * public["quality"][contenders] - .025 * station[nearby])
        b = int(contenders[np.argmax(rank)])
    keys = _keys(public)
    anchor_a = _anchor(public, prediction, a, keys)
    anchor_b = None if b is None else _anchor(public, prediction, b, keys)

    def features(idx, anchor):
        if idx is None:
            return np.zeros(17, dtype=np.float32)
        roles = prediction["role_posterior"][idx]
        return np.r_[target[idx], 1. - prediction["null"][idx], public["quality"][idx], roles,
                     np.log1p(len(anchor.old_views)) / 4., public["features"][idx, 24:29]]

    sep = 0. if anchor_b is None else np.linalg.norm(anchor_a.point - anchor_b.point)
    arc = 0. if anchor_b is None else (anchor_b.sector - anchor_a.sector + 12.) % 24. - 12.
    context = np.asarray(np.r_[region/5., anchor_a.station/160., score[a], prediction["score"][a] >= 0.,
                              b is not None, sep/22., arc/12., np.log1p(len(indices))/8.,
                              features(a, anchor_a), features(b, anchor_b)], dtype=np.float32)
    return IdentityQuery(region, (anchor_a, anchor_b), context, float(score[a]), float(target[a]),
                         bool(prediction["score"][a] >= 0.))


def outcome_truth(state, public, labels, instance):
    """Use true labels at the scoring boundary, after freezing the public anchors."""
    positions = dict(zip(_keys(public), range(len(labels)), strict=True))
    rows = [positions[a.key] if a else None for a in state.anchors]
    observed = [int(labels[i]) if i is not None else -99 for i in rows]
    return {"hypothesis": 0 if observed[0] == 0 else 1 if observed[1] == 0 else 2,
            "same_true_target": observed == [0, 0] and bool(instance[rows[0]] == instance[rows[1]]),
            "anchor_labels": observed}


def observed_event(state, public, prediction, arrival):
    """A/B support, conflicting support, or no independent support."""
    selected = (public["frames"] >= arrival) & (public["regions"] == state.region)
    coordinates = public["points"][selected]
    station, azimuth = _view_keys(public)
    view_keys = list(zip(station[selected].tolist(), azimuth[selected].tolist(), strict=True))
    sectors = public["sectors"][selected]
    zz = prediction["embedding"][selected]
    quality = public["quality"][selected]
    role = prediction["role_posterior"][selected, 0]
    values = np.full((len(coordinates), 2), -np.inf)
    if not len(coordinates):
        return 3, (0, 0)
    for j, a in enumerate(state.anchors):
        if a is None:
            continue
        ds = np.abs(public["stations"][selected] - a.station)
        distance = np.linalg.norm(coordinates - a.point, axis=1)
        angle = np.abs((sectors - a.sector + 12.) % 24. - 12.)
        new = np.fromiter((key not in a.old_views for key in view_keys), bool, len(view_keys))
        eligible = (ds <= 3.5) & (distance <= 10.) & new & (angle >= .85)
        values[eligible, j] = (-(distance[eligible]/5.)**2 - .15*ds[eligible]
                               + .35*(zz[eligible] @ a.embedding))
    votes = [set(), set()]
    for i in range(len(coordinates)):
        j = int(values[i].argmax())
        if (values[i, j] >= -2.0 and values[i, j] > values[i, 1-j] + .20
                and float(role[i] * quality[i]) >= .29):
            votes[j].add(view_keys[i])
    count = tuple(len(v) for v in votes)
    event = 2 if all(count) else 0 if count[0] else 1 if count[1] else 3
    return event, count


def _return_time(scan, first, station, sector):
    origin = (scan.node.station, scan.node.sector)
    if origin == (station, sector):
        return 0
    proxy = SimpleNamespace(env=scan.env, spec=scan.spec, limits=scan.limits, tape=scan.tape, done=False,
                            node=SimpleNamespace(station=station, sector=sector, tick=scan.node.tick + first.ticks,
                                                 distance=scan.node.distance + first.distance_mm))
    back = plan(proxy, *origin, dwell=0)
    return None if back is None else back.ticks


def views(live, state):
    """Repeat the old sound window and compare two spatially distinct windows."""
    scan = live.scan
    a, _ = state.anchors
    station = int(np.floor(a.station + .5))
    anchor_sector = int(np.floor(a.sector + .5)) % 24
    forecast = forecasts(live, station)
    proposed = []
    for name, offset in (("repeat", 0), ("left", -4), ("right", 4), ("oblique", 7)):
        angles = [(anchor_sector + offset + delta) % 24 for delta in (0, -1, 1, -2, 2)]
        for sector in angles:
            if any(row[1] == sector for row in proposed):
                continue
            trajectory = plan(scan, station, sector, dwell=4)
            if trajectory is not None:
                back = _return_time(scan, trajectory, station, sector)
                if back is not None:
                    proposed.append((name, sector, trajectory.ticks + back))
                    break
    if len(proposed) < 2 or proposed[0][0] != "repeat":
        return []
    duration = max(row[2] for row in proposed)
    values = []
    for name, sector, elapsed in proposed:
        f = forecast[sector]
        delta = (sector - anchor_sector + 12) % 24 - 12
        features = np.asarray(np.r_[state.features, delta/12., (station - scan.node.station)/160.,
                                    duration/500., f[:6]], dtype=np.float32)
        values.append(View(name, station, sector, 4 + duration - elapsed, duration, features,
                           float(f[0] + .28*f[1] - .08*f[3])))
    return values


def fork_on_tape(live, offset):
    """Fork after the last observed tick, including the active band's state."""
    from copy import copy

    old = live.scan.tape
    tick = live.scan.node.tick
    if old.branch_tick >= 0 and tick > old.branch_tick:
        active, index = old.future, tick - old.branch_tick - 1
    else:
        active, index = old.tape, tick
    tape = copy(old)
    tape.branch_tick = tick
    tape.future = type(active).make(live.case.tape_seed + offset, tape.profile, len(old.tape.normal) - tick - 1,
                                    (active.normal[index], active.dropout[index]))
    child = live.fork()
    child.scan.tape = tape
    child.done = False
    return child


def execute_view(live, chosen, state, reader, offset=None):
    branch = fork_on_tape(live, offset) if offset is not None else live.fork()
    branch.done = False
    scan = branch.scan
    start_tick, start_mm = scan.node.tick, scan.node.distance
    origin = scan.node.station, scan.node.sector
    depart = plan(scan, chosen.station, chosen.sector, chosen.dwell)
    if depart is None:
        raise RuntimeError("The admitted evidence view cannot execute")
    achieved = execute(scan, depart)
    public = scan.prefix.update(scan.node)
    event, supports = observed_event(state, public, predict(public, reader, live.case.task), achieved["arrival_tick"])
    if (scan.node.station, scan.node.sector) != origin:
        back = plan(scan, *origin, dwell=0)
        if back is None:
            raise RuntimeError("The evidence view cannot rejoin its original pose")
        execute(scan, back)
    if scan.node.tick != start_tick + chosen.ticks or len(scan.node.reader.frames) != scan.node.tick:
        raise RuntimeError("Equal-time or frame bookkeeping was violated")
    branch.cache = None
    p = scan.prefix.update(scan.node)
    raw = predict(p, reader, live.case.task)
    grade = scan.grade(p, raw)[0]
    return branch, p, raw, grade, {"event": event, "unique_support": list(supports), "ticks": chosen.ticks,
                                  "distance_mm": float(scan.node.distance - start_mm), "view": chosen.name,
                                  "station": chosen.station, "sector": chosen.sector}


def update_belief(prior, likelihood, event):
    mass = prior * likelihood[:, event]
    return mass / max(mass.sum(), 1.e-12)


def information(prior, likelihood):
    # Expected gain in Bayes identity accuracy, including the null hypothesis.
    return float(np.max(prior[:, None] * likelihood, axis=0).sum() - prior.max())


def resolve(state, public, prediction, posterior, trust):
    """Re-evaluate the saved old images using new evidence from this branch."""
    winner = int(np.argmax(posterior))
    a = state.anchors[winner] if winner < 2 else None
    if a is None or posterior[winner] < trust or winner != 1:
        return prediction
    lookup = {k: i for i, k in enumerate(_keys(public))}
    if a.key not in lookup:
        raise RuntimeError("The stored candidate address was lost")
    score = prediction["score"].copy()
    score[public["regions"] == state.region] = -1.
    score[lookup[a.key]] = posterior[winner]
    return prediction | {"score": score}


def labels_for(live, state, public):
    truth = truth_at(live.scan.node, public, live.scan.tape)
    return outcome_truth(state, public, truth["labels"], truth["instance"])
