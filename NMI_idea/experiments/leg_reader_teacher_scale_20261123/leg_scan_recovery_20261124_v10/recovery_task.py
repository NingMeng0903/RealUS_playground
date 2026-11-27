"""Material goals and public task state for residual leg-scan recovery."""

from collections import Counter
from copy import copy
from dataclasses import dataclass, replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
V9 = ROOT.parent / "leg_scan_full_route_20261124_v9"
sys.path.insert(1, str(V9))

from study_v9 import DEFAULT_PACK, ROUTES, roster as v9_roster
from scan_v9 import Scan, initialize as initialize_v9
from study_data import complete
from study_worlds import motion, truth_at


ROSTER = {key: 7_531_100_000_000_000 + 5_000_000_000 * j
          for j, key in enumerate(("smoke", "strong", "reader_train", "reader_cal", "reader_dev",
                                   "compose_teacher", "adapt", "online", "dev", "test"))}
SEARCH_TAPES = (15_000_000_001, 16_000_000_001)
TEACH_PROOF = 17_000_000_001
EVAL_TAPES = (None, 21_000_000_001, 22_000_000_001)
MAX_ACTIONS = 10
TAPE_SAMPLES = 8192
SKILLS = ("advance", "submit", "recover", "disambiguate", "revisit")
_BUNDLE = None


def roster(stage, count, round_id=0):
    source = "test" if stage == "test" else "dev" if stage in ("dev", "reader_dev") else "train"
    for j, c in enumerate(v9_roster(source, count, round_id, full=(count * 2) // 3)):
        seed = ROSTER[stage] + 100_000 * round_id + 31 * j
        yield replace(c, identifier=f"v10_{stage}_{round_id:02d}_{j:05d}", split=f"v10_{stage}",
                      world_seed=seed, tape_seed=seed + 1, action_seed=seed + 2, challenge=False)


def initialize(atlas, reader_path):
    global _BUNDLE
    from recovery_reader import load_bundle
    initialize_v9(atlas, None, None)
    _BUNDLE = load_bundle(reader_path)


@dataclass(frozen=True)
class Goal:
    skill: str
    region: int
    station: int
    sector: int = -1
    dwell: int = 0
    sweep: float = 0.

    @property
    def is_repair(self):
        return self.skill in SKILLS[2:]


class Live:
    def __init__(self, case, tape_offset=None, *, use_dird=False):
        self.scan = Scan(case, "new")
        # Extend the simulator stream without changing the already acquired
        # first frame or imposing a v9 intervention time/distance cap.
        trial = self.scan.tape
        tape = trial.tape
        extension = type(tape).make(case.tape_seed + 877_313, case.profile, TAPE_SAMPLES - len(tape.normal),
                                    (tape.normal[-1], tape.dropout[-1]))
        trial.tape = type(tape)(*(np.concatenate((getattr(tape, key), getattr(extension, key)), axis=0)
                                 for key in ("normal", "segmentation", "dropout", "false")))
        trial.future = trial.tape
        self.scan.limits = replace(self.scan.limits, ticks=TAPE_SAMPLES - 1, mm=np.inf)
        self.case = case
        self.frontier = self.scan.node.station
        self.ends = np.array([np.flatnonzero(self.scan.spec[4] == i)[-1] + self.scan.spec[1] for i in range(6)])
        self.use_dird = use_dird
        self.visited = frozenset()
        self.repair_counts = np.zeros(6, dtype=np.int16)
        self.trace = ()
        self.cache = None
        self.done = False
        if tape_offset is not None:
            tape = self.scan.tape
            self.scan.tape = tape.branch(self.scan.node.tick, self.case.tape_seed + tape_offset)

    def fork(self, tape_offset=None):
        forked = copy(self)
        tape = self.scan.tape
        if tape_offset is not None:
            tape = tape.branch(self.scan.node.tick, self.case.tape_seed + tape_offset)
        forked.scan = self.scan.fork(tape)
        forked.repair_counts = self.repair_counts.copy()
        # The observation cache only contains data up to the shared prefix.
        return forked

    def forward(self, station):
        old = self.scan.node.station
        if station > old:
            complete(self.scan.env, self.scan.spec, self.scan.node, self.scan.tape, self.scan.limits, station)
        self.frontier = max(self.frontier, self.scan.node.station)
        self.cache = None

    def get(self):
        from recovery_reader import predict
        if self.cache is None:
            p = self.scan.prefix.update(self.scan.node)
            out = predict(p, _BUNDLE, self.case.task)
            by_region = evidence(p, out, self)
            self.cache = p, out, by_region
        return self.cache

    def grade(self):
        p, predicted, _ = self.get()
        return self.scan.grade(p, predicted)[0]

    def snapshot(self, split, scope):
        p, _, _ = self.get()
        labels = truth_at(self.scan.node, p, self.scan.tape)
        return dict(public=p, labels=labels["labels"], instance=labels["instance"], layout_id=self.scan.tape.layout.identifier,
                    route=self.case.task, split=split, scope=scope, station=int(self.scan.node.station),
                    tick=int(self.scan.node.tick), trajectory_id=self.case.identifier,
                    complete=self.scan.node.station == self.scan.spec[2])

    def base_goal(self):
        station = self.scan.node.station
        end = self.scan.spec[2]
        if station == end:
            return Goal("submit", 5, end)
        if station < self.frontier:
            stop = self.frontier
        else:
            stop = next(end_i for end_i in self.ends if end_i > station)
        return Goal("advance", motion.region_of(self.scan.spec, stop), int(stop))

    def finish(self):
        self.forward(self.scan.spec[2])
        self.done = True
        return self.grade()

    def take(self, goal, trajectory=None):
        if self.done:
            raise ValueError("A submitted scan has no next action")
        from recovery_motion import execute, plan
        scan = self.scan
        station, sector, tick, mm = scan.node.station, scan.node.sector, scan.node.tick, scan.node.distance
        detail = {}
        if goal.skill == "submit":
            if station != scan.spec[2]:
                raise ValueError("Submission requires the legal terminal station")
            self.done = True
        elif goal.skill == "advance":
            self.forward(goal.station)
        elif goal.is_repair:
            trajectory = trajectory or plan(scan, goal.station, goal.sector, goal.dwell, goal.sweep, use_dird=self.use_dird)
            if trajectory is None:
                return None
            detail = execute(scan, trajectory)
            if (scan.node.station, scan.node.sector) != (goal.station, goal.sector):
                raise RuntimeError("A recovery failed to reach its material goal")
            if motion.region_of(scan.spec, scan.node.station) != goal.region:
                raise RuntimeError("Region target differs from the executed material goal")
            self.visited |= {(goal.region, goal.station, goal.sector)}
            self.repair_counts[goal.region] += 1
            self.frontier = max(self.frontier, scan.node.station)
            self.cache = None
        else:
            raise ValueError(f"Unknown recovery skill {goal.skill!r}")
        if len(scan.node.reader.frames) != scan.node.tick or (goal.skill != "submit" and tick >= scan.node.tick):
            raise RuntimeError("Motion did not advance the physical sample clock")
        result = {
            "skill": goal.skill, "goal_region": int(goal.region) + 1, "goal_station": int(goal.station),
            "goal_sector": int(goal.sector), "dwell": int(goal.dwell), "sweep": float(goal.sweep),
            "station": int(station), "sector": int(sector), "tick": int(tick), "next_station": int(scan.node.station),
            "next_sector": int(scan.node.sector), "next_tick": int(scan.node.tick), "ticks": int(scan.node.tick - tick),
            "distance_mm": float(scan.node.distance - mm), "frontier": int(self.frontier),
            "earlier_region": bool(goal.region < motion.region_of(scan.spec, station)), "dird": bool(detail.get("use_dird", False)),
            "executed_sweep": detail.get("sweep"),
        }
        self.trace += (result,)
        return result


def evidence(public, predicted, live):
    """Distinct material support and the best rival at a different observed object."""
    station = np.asarray(public["stations"])
    index = np.rint(station).astype(int)
    radius = np.asarray(public["points"])
    target = np.asarray(predicted["target"])
    score = np.asarray(predicted["winner_score"])
    submitted = np.asarray(predicted["score"])
    named = predicted.get("role_posterior")
    fallback = np.asarray(predicted["role"])
    profiles = []
    for r in range(6):
        members = np.flatnonzero(public["regions"] == r)
        expected = np.zeros(live.ends[r] - (live.ends[r-1] if r else live.scan.spec[1] - 1), dtype=float)
        first = live.ends[r] - len(expected) + 1
        peak = 0.
        good = False
        material = float(first + (len(expected) - 1) / 2.)
        rival = 0.
        artifact = 0.
        best_t = 0.
        rival_roles = np.zeros(5, dtype=float)
        if len(members):
            i = members[int(np.argmax(score[members]))]
            if score[i] < 0.:
                i = members[int(np.argmax(target[members]))]
            material = float(station[i])
            peak = max(0., float(score[i]))
            good = bool(np.any(submitted[members] >= 0.))
            best_t = float(target[i])
            local = np.clip(index[members] - first, 0, len(expected) - 1)
            np.maximum.at(expected, local, target[members])
            # Nearby same-location repeats do not become a second identity.
            gap = np.linalg.norm(radius[members] - radius[i], axis=1)
            different = (gap >= 3.5) & (np.abs(station[members] - material) <= 4.) & (members != i)
            others = members[different]
            if len(others):
                if named is not None:
                    rival_roles = np.max(named[others, 1:6], axis=0)
                    rival = float(np.max(rival_roles * .65 + np.max(target[others]) * .35))
                else:
                    rival_roles[0] = float(np.max(fallback[others, 1]))
                    rival = float(np.max(.55 * fallback[others, 1] + .45 * target[others]))
            artifact = float(named[i, 6:].sum() if named is not None else fallback[i, 2])
        coverage = float((expected >= .50).mean())
        profiles.append({"peak": peak, "accepted": good, "coverage": coverage, "expected": expected,
                         "first": int(first), "material": material, "rival": rival, "artifact": artifact,
                         "target": best_t, "roles": rival_roles})
    return profiles


def forecasts(live, station):
    scan = live.scan
    reader = scan.node.reader
    return scan.env.route.predictive(reader.model, station) @ reader.w


def _region_rank(x, current, r):
    return (1.6 * (not x["accepted"]) + .9 * (1. - x["peak"]) + .5 * (1. - x["coverage"])
            + .3 * x["rival"] - .04 * (current - r))


def _pending(live, region, state):
    """Keep an accepted regular view and reopen only an unresolved task."""
    if not state["accepted"]:
        return True
    regular = region < (5 if live.case.task == "femoral" else 3)
    if regular:
        return False
    return state["target"] < .65 or state["peak"] < .86


def _features(live, goal, trajectory, region_state, forecast):
    scan = live.scan
    s = scan.node.station
    a = scan.node.sector
    span = max(1, scan.spec[2] - scan.spec[1])
    x = region_state[goal.region]
    d = 0. if goal.sector < 0 else (goal.sector - a + 12.) % 24. - 12.
    moves = [p["coverage"] for p in region_state] + [p["peak"] for p in region_state]
    moves += [float(p["accepted"]) for p in region_state]
    moves += [p["rival"] for p in region_state]
    ticks, mm = (0., 0.) if trajectory is None else (trajectory.ticks, trajectory.distance_mm)
    feats = [
        float(live.case.task != "femoral"), (s - scan.spec[1]) / span, (live.frontier - scan.spec[1]) / span,
        np.log1p(scan.node.tick) / 7., np.log1p(scan.node.distance) / 8., np.log1p(live.repair_counts.sum()),
        goal.region / 5., (goal.station - scan.spec[1]) / span, (goal.station - s) / span,
        d / 12., goal.dwell / 8., goal.sweep, float(ticks) / 80., float(mm) / 160.,
        x["coverage"], x["target"], x["peak"], float(x["accepted"]), x["rival"], x["artifact"],
        (x["material"] - goal.station) / max(1., len(x["expected"])), np.log1p(live.repair_counts[goal.region]),
        *x["roles"], *moves, *forecast[:4], *(float(goal.skill == skill) for skill in SKILLS),
    ]
    return np.asarray(feats, dtype=np.float32)


FEATURE_DIM = 60


def options(live, focus=None):
    from recovery_motion import plan
    _, _, profiles = live.get()
    scan = live.scan
    base = live.base_goal()
    choices = [(base, None, _features(live, base, None, profiles, np.zeros(4)))]
    recent = motion.region_of(scan.spec, live.frontier)
    finished = [r for r in range(recent + 1) if live.frontier >= live.ends[r]]
    # The regular scan supplies the initial covered section. The actor can
    # target any weak, visited focus region from the current physical pose.
    first_review = 4 if live.case.task == "femoral" else 2
    allowed = finished if live.frontier >= live.ends[first_review] else []
    # The preliminary reader has a different score scale and generated the
    # hard-example deck. Only the deployed reader uses readiness semantics.
    if _BUNDLE["format"] == "recovery_public_candidate_reader_v10":
        allowed = [r for r in allowed if _pending(live, r, profiles[r])]
    elif focus is not None and focus in finished and focus not in allowed:
        allowed.append(focus)
    ranks = sorted(allowed, key=lambda r: (r == focus, _region_rank(profiles[r], recent, r)), reverse=True)
    for r in ranks[:3]:
        x = profiles[r]
        size = len(x["expected"])
        station0 = scan.node.station
        a0 = scan.node.sector
        local_m = int(round(x["material"] - x["first"]))
        samples = np.unique(np.clip(np.r_[0, size - 1, size // 2, 3 * size // 4, local_m, local_m - 1, local_m + 2,
                                          station0 - x["first"]], 0, size - 1))
        if x["accepted"]:
            samples = np.unique(np.r_[samples, int(np.argmin(x["expected"]))])
        pose_options = []
        for j in samples:
            station = x["first"] + int(j)
            # The public viability field and independent sector offsets are
            # evaluated before looking at any resulting echoes.
            table = forecasts(live, station)
            for angle in np.flatnonzero(scan.env.scanner.valid[station]):
                d = (int(angle) - a0 + 12) % 24 - 12
                rank = (table[angle, 0] + .28 * table[angle, 1] - .08 * table[angle, 3]
                        - .006 * abs(d) - .002 * abs(station0 - station) - .12 * x["expected"][j])
                pose_options.append((float(rank), station, int(angle), int(d), table[angle]))
        if not pose_options:
            continue
        pose_options.sort(key=lambda row: row[0], reverse=True)
        variants = (
            ("recover", 4, 0., lambda pose: True),
            ("disambiguate", 2, .76, lambda pose: pose[3] > 0),
            ("revisit", 3, -.76, lambda pose: pose[3] < 0 or abs(pose[1] - station0) >= 2),
        )
        added = set()
        for kind, dwell, sweep, preferred in variants:
            for _, station, sector, _, f in (p for p in pose_options if preferred(p)):
                key = (r, station, sector)
                if key in live.visited or key in added:
                    continue
                shaped = float(sweep * (.60 + .40 * np.clip(f[1], 0., 1.)))
                traj = plan(scan, station, sector, dwell, shaped, use_dird=live.use_dird)
                if traj is None:
                    continue
                goal = Goal(kind, r, station, sector, dwell, shaped)
                choices.append((goal, traj, _features(live, goal, traj, profiles, f)))
                added.add(key)
                break
    return choices


def table(choices):
    x = np.zeros((MAX_ACTIONS, FEATURE_DIM), dtype=np.float32)
    mask = np.zeros(MAX_ACTIONS, dtype=bool)
    for i, (_, _, feats) in enumerate(choices):
        x[i], mask[i] = feats, True
    return x, mask


def utility(grade, focus=None):
    r = grade["selected_ids"]
    target = 0. if focus is None else .12 * (r[focus] == 0) + .11 * grade["capture"][focus]
    return float(2.5 * grade["strict"] + grade["auxiliary"] + .055 * sum(grade["capture"]) + target)


def stats(live, grade):
    trace = list(live.trace)
    return {**grade, "steps": len(trace), "actions": dict(Counter(t["skill"] for t in trace)),
            "repairs": int(live.repair_counts.sum()), "recovery_regions": (np.flatnonzero(live.repair_counts) + 1).tolist(),
            "earlier_region_visits": sum(t["earlier_region"] for t in trace),
            "negative_station_moves": sum(t["next_station"] < t["station"] for t in trace), "trace": trace}
