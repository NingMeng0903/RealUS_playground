"""Shared training and evaluation environment for sequential material-chart scans."""

from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import pairwise

import numpy as np

import motion
import windows
from material_motion import MATERIAL_FEATURE_NAMES, MaterialMotion
from online_reader import PUBLIC_HISTORY_NAMES, PublicHistoryReader
from scan_kernel import DEPTH_MAX, DEPTH_MIN, LATERAL, PROFILES, TASKS

WIDTH = 27.
EXTRA_TICKS, EXTRA_MM = 144, 330.
ACTION_TICKS, ACTION_MM = 30, 86.
TRAIN_SHAPES = (0, 1, 2, 5)
DEV_SHAPES = (3, 4, 6)
SOURCES = ("BodyParts3D right", "BodyParts3D mirrored left", "Open3DModel right", "MedSeg")
TRAIN_SEEDS = (276_382_173, 721_423_109, 543_275_819)
DEV_SEED, SMOKE_SEED, TEST_SEED = 581_130_575, 744_025_329, 983_179_437
SKILLS = ("continue", "window", "revisit")
STAGES = ("anchor", "stop")
FEATURE_NAMES = (
    "public_detection", "public_separation", "public_detection_sd", "public_shadow",
    "public_in_window", "public_merge_risk", "belief_peak", "belief_entropy", "atlas_mode_1",
    "atlas_mode_2", "atlas_mode_3", "last_quality", "last_peak", "last_runner_up",
    "last_unknown", "last_mixed", "last_echo_count", "last_weighted_lateral",
    "last_weighted_depth", *PUBLIC_HISTORY_NAMES, "within_region_remaining", "route_remaining",
    "horizon_next_window", "horizon_next_separation", "horizon_next_dispersion",
    "available_probe_clock", "available_probe_distance", "remaining_clock", "remaining_distance",
    "mainline_reserve", "mainline_distance_reserve", "stop_stage", "stop_elapsed",
    "stop_return_reserve", "stop_remaining", "stop_elapsed_distance", "stop_return_distance",
    "stop_remaining_distance", "stop_parent_window", "stop_parent_revisit", *MATERIAL_FEATURE_NAMES,
)


@dataclass(frozen=True)
class Case:
    identifier: str
    split: str
    source: int
    shape: int
    task: str
    profile: str
    world_seed: int
    tape_seed: int
    action_seed: int
    band: int
    challenge: bool = False


@dataclass(frozen=True)
class Arm:
    name: str
    policy: str = ""
    sampled: bool = False
    feedback: bool = True
    stop: bool = True
    fixed_control: bool = False
    event_timing: bool = False
    position: bool = True
    dird: bool = True


def run_cases(count, split, seed, start=0):
    shape_set = DEV_SHAPES if split in ("calibration", "development") else TRAIN_SHAPES
    tasks = tuple(TASKS)
    first, last = start // 4, (start + count + 3) // 4
    for group in range(first, last):
        rng = np.random.default_rng(np.random.SeedSequence([seed, group]))
        for offset, source in enumerate(rng.permutation(4)):
            i = group * 4 + offset
            shape = int(rng.choice(shape_set))
            # Each case receives a distinct, reproducible geometry/measurement/action seed.
            seeds = [int(seed) * 2**32 + 3 * i + purpose for purpose in range(3)]
            if not start <= i < start + count:
                continue
            # IDs in the shard are the authoritative count. Distinct case tapes
            # are checked across each split in the runner's preflight.
            task = "medseg_labelled_trunk" if source == 3 else tasks[(group + 2 * source) % len(tasks)]
            yield Case(f"{split}_{seed}_{i:06d}", split, int(source), shape, task,
                       PROFILES[(group + source) % 2], *seeds, group, (group + 3 * source) % 5 < 2)


def test_cases(repetitions=16):
    for band in range(repetitions):
        for task_no, task in enumerate(TASKS):
            for profile_no, profile in enumerate(PROFILES):
                i = band * 10 + task_no * 2 + profile_no
                base = TEST_SEED << 32
                yield Case(f"blender_{band:02d}_{task}_{profile}", "blender", 4, 0, task, profile,
                           base + 3 * i, base + 3 * i + 1, base + 3 * i + 2, band)


def spec_for(scanner, case):
    if case.source == 3:
        return motion.definition(0, *scanner.ct_bounds)
    hi, lo, roles = TASKS[case.task]
    indices = np.flatnonzero((scanner.stations <= hi) & (scanner.stations >= lo))
    return motion.definition(0, int(indices[0]), int(indices[-1]), roles)


class Route:
    """Common constrained MPC and fixed-rate acquisition between anchor stations."""

    deltas = np.array([0, -1, 1, -2, 2])
    sectors = (np.arange(motion.SECTORS)[:, None] + deltas) % motion.SECTORS

    def __init__(self, scanner):
        self.scanner = scanner
        self.surface = scanner.surface
        s, a = np.indices((len(scanner.stations) - 1, motion.SECTORS))
        sample = np.linspace(0., 1., 21)[None, None, None, :]
        station = np.broadcast_to(s[..., None, None] + sample, (*s.shape, len(self.deltas), 21))
        sector = a[..., None, None] + self.deltas[None, None, :, None] * sample
        coordinates, legal = self.surface.path_coordinates(station, sector)
        lengths = np.linalg.norm(np.diff(coordinates, axis=-2), axis=-1)
        self.legal = legal.all(axis=-1)
        self.distance = lengths.sum(axis=-1) + .015
        # Bound the maximum per-frame chart arc, including the surface curvature.
        self.clock = np.maximum(1, np.ceil(1.035 * lengths.max(axis=-1) * 20 /
                                          (motion.CLOCK_S * motion.MAX_SPEED_MM_S))).astype(np.int16)
        self.cache = {}

    @lru_cache(maxsize=12)
    def reserve(self, first, last):
        ticks = np.full((last - first + 1, motion.SECTORS), 100_000, dtype=np.int32)
        distance = np.full(ticks.shape, np.inf)
        ticks[-1, self.scanner.valid[last]] = 0
        distance[-1, self.scanner.valid[last]] = 0.
        for i in reversed(range(last - first)):
            s = i + first
            cost_clock = self.clock[s] + ticks[i + 1, self.sectors]
            cost_distance = self.distance[s] + distance[i + 1, self.sectors]
            score = np.where(self.legal[s], cost_clock + .06 * cost_distance, np.inf)
            best = np.argmin(score, axis=1)
            for a in np.flatnonzero(np.isfinite(score).any(axis=1)):
                ticks[i, a], distance[i, a] = cost_clock[a, best[a]], cost_distance[a, best[a]]
        return ticks, distance

    @lru_cache(maxsize=2048)
    def predictive(self, model, station):
        n = len(model.prior)
        output = np.zeros((motion.SECTORS, 6, n))
        for a in np.flatnonzero(self.scanner.valid[station]):
            output[a] = model.predictive(station, int(a))
        return output

    def bounded_candidates(self, node, spec, limits):
        s, a = int(node.station), int(node.sector)
        i = s - spec[1]
        sectors = self.sectors[a]
        return (self.legal[s, a]
                & (node.tick + self.clock[s, a] + limits.reserve_ticks[i + 1, sectors] <= limits.ticks)
                & (node.distance + self.distance[s, a] + limits.reserve_mm[i + 1, sectors] <= limits.mm + 1.e-7))

    def advance(self, node, spec, trial, limits):
        s, a = int(node.station), int(node.sector)
        if s == spec[2]:
            return
        legal = self.bounded_candidates(node, spec, limits)
        if not legal.any():
            raise RuntimeError("No budget-feasible continuation despite reserved route")
        sectors = self.sectors[a]
        model, w = node.reader.model, node.reader.w
        forecast = self.predictive(model, s + 1)[sectors] @ w
        score = forecast[:, 0] + .19 * forecast[:, 1] - .013 * self.distance[s, a]
        if s + 1 < spec[2]:
            next_detection = self.predictive(model, s + 2)[:, 0] @ w
            future = next_detection[self.sectors[sectors]] - .013 * self.distance[s + 1, sectors]
            forward = np.max(np.where(self.legal[s + 1, sectors], future, -1.), axis=1)
            score += .43 * forward
        slot = int(np.argmax(np.where(legal, score, -np.inf)))
        key = s, a, slot
        path = self.cache.get(key)
        if path is None:
            ticks = self.clock[s, a, slot]
            positions = np.column_stack((s + np.arange(ticks + 1) / ticks,
                                         a + self.deltas[slot] * np.arange(ticks + 1) / ticks))
            lengths = self.scanner._path_metrics(positions, spec)
            if lengths is None:
                raise RuntimeError(f"Sampled common path violates chart constraints at {key}")
            # Reserve the conservative grid distance. The recorded distance is
            # from the actual physically sampled path with the same subdivision.
            path = self.cache[key] = (positions[1:], lengths)
        for (station, sector), length in zip(*path):
            node.tick += 1
            node.distance += float(length)
            self.scanner._observe(node, spec, trial, float(station), float(sector), node.tick)
        node.station, node.sector = s + 1, int(sectors[slot])


@dataclass(frozen=True)
class Limits:
    ticks: int
    mm: float
    reserve_ticks: np.ndarray
    reserve_mm: np.ndarray

    @classmethod
    def build(cls, route, spec, node, scale=1.):
        ticks, mm = route.reserve(spec[1], spec[2])
        a = node.sector
        return cls(int(node.tick + ticks[0, a] + scale * EXTRA_TICKS),
                   float(node.distance + mm[0, a] + scale * EXTRA_MM), ticks, mm)

    def probe(self, node, spec):
        index, angle = int(node.station) - spec[1], int(node.sector)
        ticks = max(0, min(ACTION_TICKS, self.ticks - node.tick - self.reserve_ticks[index, angle]))
        mm = max(0., min(ACTION_MM, self.mm - node.distance - self.reserve_mm[index, angle]))
        return int(ticks), float(mm)


class ScanEnvironment:
    def __init__(self, scanner, layout_factory=None):
        self.scanner = scanner
        self.scanner.reader_type = PublicHistoryReader
        self.route = Route(scanner)
        self.material = MaterialMotion(scanner)
        self.model = scanner.model(WIDTH)
        self.layout_factory = layout_factory

    def start(self, case, spec, budget_scale=1., trial_type=motion.Trial):
        trial = trial_type(case.source, case.tape_seed, case.profile, world_seed=case.world_seed,
                           split=case.split)
        if self.layout_factory is not None:
            trial.layout = self.layout_factory.build(case.source, case.world_seed, case.profile, case.split)
        node = self.scanner.start(spec, trial, self.model)
        limits = Limits.build(self.route, spec, node, budget_scale)
        if case.challenge and trial.layout is not None:
            trial.window = windows.configure(self, node, spec, case, trial, ACTION_TICKS, ACTION_MM)
            if trial.window is not None:
                node = self.scanner.start(spec, trial, self.model)
        return node, trial, limits

    def features(self, node, spec, limits, caps, arm, stop_info=None, skill=0):
        reader = node.reader
        s, a, echoes, _, quality, region, post, unknown, _ = reader.frames[-1]
        w = reader.w
        forecast = self.model.feature(s, a, w)
        history = reader.public_summary(region)
        mixed = reader.mixed_posterior[-1] if reader.mixed_posterior else np.empty(0)
        base = min(spec[2], max(spec[1], int(s + .5)))
        anchor = base - spec[1]
        beam = int(a + .5) % motion.SECTORS
        at_next = min(spec[2], base + 1)
        windows = (self.route.predictive(self.model, at_next)[self.route.sectors[beam]] @ w)
        next_region_last = min(spec[2], spec[1] + int(np.searchsorted(spec[4], region + 1)) - 1)
        ordered = np.sort(post)
        weight = post / (.05 + post.sum())
        lateral = float(weight @ echoes[:, 0]) / LATERAL
        depth = float(weight @ echoes[:, 1]) / DEPTH_MAX
        confidence = (max(post, default=0.), ordered[-2] if len(ordered) > 1 else 0.)
        at_stop = stop_info is not None
        current_distance = node.distance
        if at_stop:
            current_distance += stop_info["elapsed_distance_mm"]
            extra = [1., stop_info["elapsed_ticks"] / ACTION_TICKS,
                     stop_info["return_reserve_ticks"] / ACTION_TICKS,
                     stop_info["remaining_ticks"] / ACTION_TICKS,
                     stop_info["elapsed_distance_mm"] / ACTION_MM,
                     stop_info["return_reserve_mm"] / ACTION_MM,
                     stop_info["remaining_mm"] / ACTION_MM, float(skill == 1), float(skill == 2)]
        else:
            extra = [0.] * 9
        material = (self.material.features(node, spec, use_dird=arm.dird)
                    if arm.position else np.zeros(len(MATERIAL_FEATURE_NAMES)))
        vector = np.r_[forecast, w.max(), -w @ np.log(np.maximum(w, 1.e-30)) / np.log(len(w)),
                       w.reshape(3, -1).sum(axis=1), quality, confidence, max(unknown, default=0.),
                       max(mixed, default=0.), len(echoes) / 7., lateral, depth, history,
                       (next_region_last - s) / max(1., (spec[2] - spec[1]) / 6.),
                       (spec[2] - s) / (spec[2] - spec[1]), windows[:, 0].max(),
                       windows[:, 1].max(), np.ptp(windows[:, 0]), caps[0] / ACTION_TICKS,
                       caps[1] / ACTION_MM, (limits.ticks - node.tick) / limits.ticks,
                       (limits.mm - current_distance) / limits.mm, limits.reserve_ticks[anchor, beam] / limits.ticks,
                       limits.reserve_mm[anchor, beam] / limits.mm, extra, material]
        if len(vector) != len(FEATURE_NAMES):
            raise RuntimeError("Public feature layout changed")
        return np.clip(np.nan_to_num(vector, nan=0., posinf=1., neginf=-1.), -2., 2.).astype(np.float32)

    def mask(self, node, spec, caps, arm):
        available = caps[0] >= 4 and caps[1] >= 2.
        if arm.event_timing and node.station < spec[2]:
            i = node.station - spec[1]
            available &= spec[4][i] != spec[4][i + 1]
        return np.array([True, available, available and node.station > spec[1]])

    def plan(self, node, spec, skill, controls, caps, arm):
        return self.material.plan_probe(node, spec, skill, controls, *caps, use_dird=arm.dird)

    def public_proposals(self, node, spec, mask, caps, arm, force=False):
        """Rank feasible actions by a public mixture of forecast and acquired evidence."""
        reader, station, sector = node.reader, int(node.station), int(node.sector)
        model, w = reader.model, reader.w
        region = motion.region_of(spec, station)
        history = reader.public_summary(region)
        current = model.feature(station, sector, w)
        normal = self.route.predictive(model, min(station + 1, spec[2]))[self.route.sectors[sector]] @ w
        ref = max(current[0], normal[:, 0].max() * .80)
        proposals = []
        for skill in (1, 2):
            if not mask[skill]:
                continue
            scored = []
            for turn, offset, shape in ((-.76, .62, -.38), (-.42, .38, .30), (.42, .38, -.30), (.76, .62, .38)):
                cmd = np.array([turn, offset, shape], dtype=np.float32)
                goal = self.material._skill_goal(np.array([station, sector], dtype=float), cmd, skill, spec)
                near, angle = int(goal[0] + .5), int(goal[1] + .5) % motion.SECTORS
                if not self.scanner.valid[near, angle]:
                    continue
                p = model.feature(near, angle, w)
                uncertainty = (1. - history[0]) * (.20 * ref + .09 * history[4])
                diversity = .075 * min(1., abs(turn) + (skill == 2) * .15)
                score = (p[0] - ref + .17 * (p[1] - current[1]) + .13 * (p[5] < current[5])
                         + .10 * (p[4] - current[4]) - .08 * p[3] + uncertainty + diversity)
                scored.append((score, cmd))
            for score, cmd in sorted(scored, key=lambda value: value[0], reverse=True):
                # Assign continuing MPC zero reference value and compare it
                # in the same maximization as every proposed probe.
                if score <= 0. and not force:
                    break
                plan = self.plan(node, spec, skill, cmd, caps, arm)
                if plan is not None:
                    proposals.append((float(score), skill, cmd, plan))
                    break
        return sorted(proposals, key=lambda value: value[0], reverse=True)


def public_choice(reader, region, threshold):
    selected = reader.select(region, threshold)
    return None if selected is None else (int(selected[0]), int(selected[1]), float(selected[2]))


def station_bins(spec, y):
    stations = np.asarray(y[spec[1]:spec[2] + 1])
    edges = np.r_[stations[0], (stations[:-1] + stations[1:]) / 2., stations[-1]]
    return abs(np.diff(edges))


class Score:
    """Evaluator-only ledger. True identities are inaccessible to the actor."""

    def __init__(self, spec, y):
        self.spec = spec
        self.bins = station_bins(spec, y)
        self.captured = np.zeros(len(spec[4]), dtype=bool)
        self.potential = np.zeros(6)
        self.cursor = 0

    def update(self, reader):
        spec = self.spec
        touched = set()
        capture_mm = 0.
        for frame in reader.frames[self.cursor:]:
            station = min(spec[2], max(spec[1], int(frame[0] + .5))) - spec[1]
            if not self.captured[station] and 0 in frame[3]:
                self.captured[station] = True
                capture_mm += self.bins[station]
            # Adjacent zone evidence can change as soon as the next view arrives.
            touched.update(spec[4][max(station - 3, 0):station + 4])
        self.cursor = len(reader.frames)
        delta = 0.
        for region in touched:
            frames, slots, evidence = reader.ranked(int(region))
            if len(evidence):
                labels = np.fromiter((reader.frames[i][3][j] == 0 for i, j in zip(frames, slots)),
                                     dtype=bool, count=len(evidence))
                maximum_true = evidence[labels].max(initial=0.)
                maximum_false = evidence[~labels].max(initial=0.)
                p = .75 * maximum_true - .37 * max(0., (maximum_false - .35) / .65) ** 2
            else:
                p = 0.
            delta += p - self.potential[region]
            self.potential[region] = p
        return float(delta), float(.8 * capture_mm / max(1., self.bins.sum()))

    def delivery(self, reader, threshold, complete):
        spec = self.spec
        picked, public_graph = [], []
        correct = np.zeros(len(self.bins), dtype=bool)
        delivered = np.zeros(len(self.bins), dtype=bool)
        edges = []
        for region in range(6):
            choice = public_choice(reader, region, threshold)
            picked.append(-1 if choice is None else int(reader.frames[choice[0]][3][choice[1]] == 0))
            if choice is not None:
                public_graph.append([region, *choice])
            frames, slots, evidence = reader.ranked(region)
            # Commit at most one hypothesis per station, using the same final rejection threshold.
            for fi, slot, strength in zip(frames, slots, evidence):
                if strength < threshold:
                    break
                station = min(spec[2], max(spec[1], int(reader.frames[fi][0] + .5))) - spec[1]
                if not delivered[station]:
                    delivered[station] = True
                    correct[station] = reader.frames[fi][3][slot] == 0
        for left, right in pairwise(public_graph):
            if right[0] != left[0] + 1:
                continue
            l, r = reader.frames[left[1]], reader.frames[right[1]]
            if abs(r[0] - l[0]) <= 3.2:
                lp, rp = reader.absolute(l[0], l[1], l[2][left[2]]), reader.absolute(r[0], r[1], r[2][right[2]])
                if np.linalg.norm(lp - rp) <= 11.:
                    edges.append([left[0], right[0]])
        capture_regions = np.unique(spec[4][self.captured]).size
        return {
            "complete": bool(complete), "pass": int(complete and picked == [1] * 6),
            "wrong_delivery": int(0 in picked), "correct_regions": picked.count(1),
            "wrong_regions": picked.count(0), "no_result_regions": picked.count(-1),
            "captured_regions": int(capture_regions), "region_outcomes": picked,
            "correct_mm": float(self.bins @ correct), "capture_mm": float(self.bins @ self.captured),
            "submitted_mm": float(self.bins @ delivered), "public_graph": public_graph, "public_edges": edges,
            "correct_station_mask": np.flatnonzero(correct).tolist(),
            "capture_station_mask": np.flatnonzero(self.captured).tolist(),
            "region_potential": self.potential.tolist(),
        }


def end_reward(metrics):
    return (2.4 * metrics["pass"] + .48 * metrics["correct_regions"]
            - .77 * metrics["wrong_regions"] - .30 * metrics["wrong_delivery"])


@dataclass
class Rollout:
    metrics: dict
    steps: list
    frames: list
    mixed: list
    probes: list
    teaching: dict | None
    audit: list | None
    layout_centers: np.ndarray | None


def _record(timeline, score, node, features, decision, mask, skill, stage, caps):
    if timeline:
        _close_step(timeline[-1], score, node, features)
    timeline.append({
        "features": features, "stage": stage, "mask": mask, "skill": int(skill),
        "latent": np.zeros(3, dtype=np.float32) if decision is None else decision.latent,
        "controls": np.zeros(3, dtype=np.float32) if decision is None else decision.action,
        "probabilities": np.eye(3, dtype=np.float32)[skill] if decision is None else decision.probabilities,
        "old_logp": 0. if decision is None else decision.log_probability,
        "value": 0. if decision is None else decision.value,
        "tick": int(node.tick), "station": float(node.station), "sector": float(node.sector),
        "first_frame": len(node.reader.frames), "tick_cap": caps[0], "mm_cap": caps[1],
        "done": False,
    })


def _close_step(row, score, node, following):
    identity, acquisition = score.update(node.reader)
    row.update({"identity_reward": identity, "capture_reward": acquisition, "terminal_reward": 0.,
                "reward": identity + acquisition, "ticks": node.tick - row["tick"],
                "frame_stop": len(node.reader.frames), "next_features": following,
                "next_tick": int(node.tick), "next_station": float(node.station),
                "next_sector": float(node.sector)})


def _fixed_controls(skill):
    return np.array([.46 if skill == 1 else -.46, .50, -.15], dtype=np.float32)


def _prefix_marks(spec, rng):
    marks = []
    for region in sorted(rng.choice(6, 3, replace=False)):
        candidates = np.flatnonzero(spec[4] == region) + spec[1]
        marks.append(int(rng.choice(candidates)))
    return set(marks)


def _public_stop(reader, prior, info, region):
    history = reader.public_summary(region)
    elapsed, reserve = info["elapsed_ticks"], info["return_reserve_ticks"]
    current = history[0] - prior[0] + .19 * (history[3] - prior[3])
    return current > .01 and reserve <= elapsed + 2.


def rollout(env, case, arm, threshold, actor=None, *, teacher=False, audit=False, forced=False,
            budget_scale=1., trial_type=motion.Trial, diagnostics_ids=False):
    spec = spec_for(env.scanner, case)
    node, trial, limits = env.start(case, spec, budget_scale, trial_type)
    rng = np.random.default_rng(case.action_seed)
    teacher_rng = np.random.default_rng(np.random.SeedSequence([case.action_seed, 414]))
    score = Score(spec, env.scanner.surface.y)
    score.update(node.reader)
    steps, probes, prefixes = [], [], []
    marks = _prefix_marks(spec, teacher_rng) if teacher else set()
    attempted = [0, 0, 0]
    succeeded = [0, 0, 0]
    anchor_count = 1
    total_probe_ticks, total_probe_mm, stop_count, possible = 0, 0., 0, 0
    for _ in range(spec[2] - spec[1] + 1):
        caps = limits.probe(node, spec)
        mask = env.mask(node, spec, caps, arm)
        features = env.features(node, spec, limits, caps, arm)
        plan = decision = None
        controls = np.zeros(3, dtype=np.float32)
        skill = 0
        if forced and mask[1]:
            first = min(4, spec[2] - spec[1])
            route_index = int(node.station) - spec[1]
            if route_index in (1, first):
                skill = 1 if route_index == 1 else 2
                turn = .44 if node.sector < motion.SECTORS // 2 else -.44
                controls = np.array([turn, .53, .42], dtype=np.float32)
        elif actor is not None:
            decision = actor.choose(features, mask, rng if arm.sampled else None)
            skill, controls = decision.skill, decision.action
            if skill and arm.fixed_control:
                controls = _fixed_controls(skill)
        elif arm.name == "active_mpc" and mask[1]:
            options = env.public_proposals(node, spec, mask, caps, arm)
            if options:
                _, skill, controls, plan = options[0]
        possible += int(mask[1])
        _record(steps, score, node, features, decision, mask, skill, 0, caps)
        steps[-1]["controls"] = controls
        if node.station in marks and mask[1]:
            prefixes.append((node.fork(), features.copy(), mask.copy(), caps, skill, controls.copy()))
        if skill:
            attempted[skill] += 1
            region = motion.region_of(spec, node.station)
            start_station, start_sector = node.station, node.sector
            start_frame, start_tick = len(node.reader.frames), node.tick
            prior = node.reader.public_summary(region)
            high_index = len(steps) - 1

            def stop_here(in_motion, info):
                stage_features = env.features(in_motion, spec, limits, caps, arm, info, skill)
                stop_decision = actor.stop(stage_features, rng if arm.sampled else None) if actor is not None else None
                action = (int(forced) if forced else (stop_decision.skill if stop_decision is not None
                          else int(_public_stop(in_motion.reader, prior, info, region))))
                _record(steps, score, in_motion, stage_features, stop_decision,
                        np.array([True, True, False]), action, 1, caps)
                steps[-1]["exit_check"] = info.copy()
                return action == 1

            if plan is None:
                plan = env.plan(node, spec, skill, controls, caps, arm)
            event = env.material.run_probe(node, spec, trial, plan, *caps, feedback=arm.feedback,
                                           stop=stop_here if arm.stop else None, use_dird=arm.dird,
                                           copy_node=False)
            if event.legal:
                node = event.node
                if (int(node.station), int(node.sector)) != (start_station, start_sector):
                    raise RuntimeError("Probe did not physically rejoin its reserved route anchor")
                if not np.allclose(event.trajectory[-1, 1:3], [start_station, start_sector], atol=1.e-6):
                    raise RuntimeError("Actual final probe frame disagrees with its route anchor")
                if len(node.reader.frames) - start_frame != event.ticks:
                    raise RuntimeError("Probe dropped acquisition frames")
                succeeded[skill] += 1
                total_probe_ticks += event.ticks
                total_probe_mm += event.distance
                stop_count += int(event.stopped)
            elif event.ticks or event.node is not node:
                raise RuntimeError(f"Probe failed after consuming controls: {event.reason}")
            trace = event.trajectory
            axial = motion._between(env.scanner.surface.y, start_station)
            movement = float(np.max(np.abs(np.interp(trace[:, 1], np.arange(len(env.scanner.stations)),
                                                      env.scanner.surface.y) - axial))) if len(trace) else 0.
            circumference = float(np.max(np.abs(motion._sector_delta(start_sector, trace[:, 2])))
                                  * motion.ANGLE_STEP) if len(trace) else 0.
            probes.append({"step": high_index, "skill": skill, "controls": controls,
                           "station": start_station, "sector": start_sector, "region": region,
                           "tick": start_tick, "frame_start": start_frame, "legal": event.legal,
                           "actual_axial_mm": movement, "actual_circumferential_rad": circumference,
                           "ticks": event.ticks, "distance_mm": event.distance, "stopped": event.stopped,
                           "feedback_updates": event.feedback_updates, "fraction": event.applied_fraction,
                           "opportunities": len(event.stop_opportunities), "reason": event.reason,
                           "dird_applied": bool(event.plan is not None and
                                                event.plan.dird_metadata.get("chart_score_applied", False)),
                           "goal": None if event.plan is None else event.plan.goal,
                           "exit": event.exit,
                           "before_region_end": bool(start_station <
                               spec[1] + int(np.searchsorted(spec[4], region + 1)) - 1)})
        if node.station == spec[2]:
            break
        env.route.advance(node, spec, trial, limits)
        anchor_count += 1
        if node.tick > limits.ticks or node.distance > limits.mm + 1.e-6:
            raise RuntimeError("Route exceeded its shared clock or distance budget")
    _close_step(steps[-1], score, node, env.features(node, spec, limits, (0, 0.), arm))
    complete = node.station == spec[2] and anchor_count == len(spec[4])
    metrics = score.delivery(node.reader, threshold, complete)
    steps[-1]["done"] = complete
    steps[-1]["terminal_reward"] = end_reward(metrics)
    steps[-1]["reward"] += steps[-1]["terminal_reward"]
    if len(node.reader.frames) != node.tick or not complete:
        raise RuntimeError(f"Incomplete or unsynchronized episode {case.identifier}")
    metrics.update({
        "attempted": attempted, "succeeded": succeeded, "probe_ticks": total_probe_ticks,
        "probe_mm": total_probe_mm, "proposed": sum(attempted), "probes": sum(succeeded),
        "early_stops": stop_count, "stop_opportunities": sum(p["opportunities"] for p in probes),
        "feedback_updates": sum(p["feedback_updates"] for p in probes),
        "dird_probes": sum(p["dird_applied"] for p in probes), "early_probes":
            sum(p["legal"] and p["before_region_end"] for p in probes), "available_anchors": possible,
        "anchors": anchor_count, "end_tick": node.tick, "time_s": node.tick * motion.CLOCK_S,
        "travel_mm": float(node.distance), "cap_ticks": limits.ticks, "cap_mm": limits.mm,
        "frames": len(node.reader.frames), "steps": len(steps), "reward": sum(s["reward"] for s in steps),
        "route_length_mm": float(score.bins.sum()), "last_station": int(node.station),
    })
    if trial.layout is not None:
        metrics["layout_id"] = trial.layout.identifier
        metrics["layout"] = trial.layout.metadata
    if trial.window is not None:
        metrics["recoverable_window"] = asdict(trial.window)
        metrics["acquired_in_ambiguous_window"] = sum(
            abs(f[0] - trial.window.station) < 1. and
            abs(motion._sector_delta(trial.window.sector, f[1])) < 1.
            for f in node.reader.frames)
    annotation = (teacher_annotation(env, spec, trial, prefixes, metrics, arm, actor, threshold,
                                     case, teacher_rng, limits) if teacher else None)
    ranks = reader_audit(node.reader) if audit else None
    if diagnostics_ids:
        metrics["evidence"] = [node.reader.explanation(region) for region in range(6)]
    return Rollout(metrics, steps, node.reader.frames, node.reader.mixed_posterior, probes, annotation, ranks,
                   None if trial.layout is None else trial.layout.centers)


def reader_audit(reader):
    result = []
    for region in range(6):
        frames, slots, scores = reader.ranked(region)
        truth = np.array([reader.frames[fi][3][slot] == 0 for fi, slot in zip(frames, slots)], dtype=bool)
        old_best = None
        opts = reader.options(region)
        for fi, slot, s, a, point, probability, quality in opts:
            if probability < .12:
                continue
            support = max((other[5] * np.exp(-.5 * (np.linalg.norm(other[4] - point) / 5.5) ** 2)
                           for other in opts if other[5] >= .12 and
                           (.65 <= abs(other[2] - s) <= 3. or
                            (abs(other[2] - s) < .25 and abs(motion._sector_delta(a, other[3])) >= 2.))),
                          default=0.)
            total = .75 * probability + .22 * support + .03 * quality
            if old_best is None or total > old_best[0]:
                old_best = float(total), bool(reader.frames[fi][3][slot] == 0), float(probability)
        result.append({"region": region, "candidates": len(scores), "true_seen": bool(truth.any()),
                       "true_rank": None if not truth.any() else int(np.flatnonzero(truth)[0]) + 1,
                       "best": None if not len(scores) else [float(scores[0]), bool(truth[0])],
                       "old": old_best,
                       "old_commit": bool(old_best and old_best[0] >= .66 and old_best[2] >= .55)})
    return result


def _replay_score(node, spec, region, threshold, first):
    fi, slot, evidence = node.reader.ranked(region)
    true = np.array([node.reader.frames[i][3][j] == 0 for i, j in zip(fi, slot)], dtype=bool)
    p_true = float(evidence[true].max(initial=0.))
    p_false = float(evidence[~true].max(initial=0.))
    choice = public_choice(node.reader, region, threshold)
    outcome = -1 if choice is None else int(node.reader.frames[choice[0]][3][choice[1]] == 0)
    existing, gained = set(), set()
    for i, frame in enumerate(node.reader.frames):
        if frame[5] == region and 0 in frame[3]:
            (existing if i < first else gained).add(int(frame[0] + .5))
    capture = len(gained - existing)
    soft = .75 * p_true - .37 * max(0., (p_false - .35) / .65) ** 2
    utility = (1.20 * (outcome == 1) - 1.55 * (outcome == 0)
               + .70 * soft + .10 * min(capture, 4))
    return {"utility": utility, "outcome": outcome, "new_true_stations": capture,
            "true_evidence": p_true, "wrong_evidence": p_false, "tick": node.tick}


def teacher_annotation(env, spec, trial, prefixes, actual, arm, actor, threshold, case, rng, limits):
    output = {"eligible_prefix": bool(prefixes), "branches": []}
    if not prefixes:
        return output
    success = rng.random() < .23
    selected = [p for p in prefixes if (actual["region_outcomes"][motion.region_of(spec, p[0].station)] == 1) == success]
    prefix, features, mask, caps, skill, controls = (selected or prefixes)[int(rng.integers(len(selected or prefixes)))]
    region = motion.region_of(spec, prefix.station)
    future_seed = 8_000_000_000_000_000_000 + case.tape_seed
    tapes = (trial, trial.branch(prefix.tick, future_seed))
    assert tapes[0].layout is tapes[1].layout and tapes[0].window is tapes[1].window
    options = [("continue", 0, np.zeros(3), None), ("student", skill, controls, None)]
    for _, action, params, plan in env.public_proposals(prefix, spec, mask, caps, arm, force=True):
        options.append(("public_search", action, params, plan))
    for action in (1, 2):
        if mask[action]:
            options.append(("continuous_draw", action,
                            rng.uniform([-.86, -.82, -.74], [.86, .82, .74]), None))
    horizon = min(spec[2], int(prefix.station) + 4)
    nodes = []
    outcomes = []
    for origin, action, params, plan in options:
        if plan is None and action:
            plan = env.plan(prefix, spec, action, params, caps, arm)
        feasible = not action or plan is not None
        pair, pair_nodes = [], []
        for future in tapes:
            node = prefix.fork()
            ticks, applied = 0, 0.
            if action and feasible:
                def replay_stop(n, info):
                    f = env.features(n, spec, limits, caps, arm, info, action)
                    return bool(actor.stop(f).skill)
                event = env.material.run_probe(node, spec, future, plan, *caps, feedback=arm.feedback,
                                               stop=replay_stop if arm.stop else None, use_dird=arm.dird)
                if not event.legal:
                    raise RuntimeError("Teacher verified plan failed under a measurement branch")
                node, ticks, applied = event.node, event.ticks, event.applied_fraction
            while node.station < horizon:
                env.route.advance(node, spec, future, limits)
            pair.append(_replay_score(node, spec, region, threshold, len(prefix.reader.frames)) |
                        {"probe_ticks": ticks, "applied_fraction": applied})
            pair_nodes.append(node)
        nodes.append(pair_nodes)
        outcomes.append({"origin": origin, "skill": action, "controls": params, "feasible": feasible,
                         "replays": pair})
    for row in outcomes:
        row["paired_gain"] = [item["utility"] - baseline["utility"]
                              for item, baseline in zip(row["replays"], outcomes[0]["replays"])]
    best = max((i for i, branch in enumerate(outcomes) if branch["feasible"]),
               key=lambda i: outcomes[i]["paired_gain"][0])
    full_tail = None
    if rng.random() < .20 and best != 0:
        full_tail = {"candidate": best, "paired_terminal_gain": [], "terminal": []}
        for j, tape in enumerate(tapes):
            utility, summaries = [], []
            for i in (0, best):
                node = nodes[i][j]
                while node.station < spec[2]:
                    env.route.advance(node, spec, tape, limits)
                score = Score(spec, env.scanner.surface.y)
                score.update(node.reader)
                metrics = score.delivery(node.reader, threshold, node.station == spec[2])
                value = end_reward(metrics) + sum(metrics["region_potential"]) + .8 * metrics["capture_mm"] / score.bins.sum()
                utility.append(value)
                summaries.append({key: metrics[key] for key in ("pass", "correct_regions", "wrong_regions", "correct_mm")})
            full_tail["paired_terminal_gain"].append(float(utility[1] - utility[0]))
            full_tail["terminal"].append(summaries)
        full_tail["sign_conflict"] = [
            bool(np.sign(gain) != np.sign(full)) for gain, full in
            zip(outcomes[best]["paired_gain"], full_tail["paired_terminal_gain"])]
    return output | {"features": features, "mask": mask, "branches": outcomes, "stage": 0,
                     "requested_success_control": bool(success), "student_outcome": actual["region_outcomes"][region],
                     "prefix_station": int(prefix.station), "prefix_sector": int(prefix.sector),
                     "prefix_region": region, "prefix_frames": len(prefix.reader.frames), "prefix_tick": int(prefix.tick),
                     "local_horizon": horizon, "student_tape_seed": case.tape_seed,
                     "independent_tape_seed": future_seed, "full_tail": full_tail,
                     "fixed_layout_id": None if trial.layout is None else trial.layout.identifier}


def physical_subset(env, spec, rollout_metrics, source):
    """An evaluator-only geometric observability ceiling for the raw target label."""
    if not hasattr(env, "physical_cache"):
        env.physical_cache = {}
    key = (spec[1], spec[2])
    if key not in env.physical_cache:
        good, present, inside = [], [], []
        for station in range(spec[1], spec[2] + 1):
            bone = env.scanner._anatomy
            present.append(bool(np.isfinite(bone.centers[0, station, 0]).all()))
            inside.append(bool(bone.inside[station, 0]))
            visible = False
            for angle in np.flatnonzero(env.scanner.valid[station]):
                points, gates = bone.truth(env.scanner.surface, source, station, int(angle))
                lateral, depth = points[0]
                visible |= bool(-LATERAL <= lateral <= LATERAL and DEPTH_MIN <= depth <= DEPTH_MAX and gates[0] < .95)
            good.append(visible)
        bins = station_bins(spec, env.scanner.surface.y)
        env.physical_cache[key] = np.asarray(good), np.asarray(present), np.asarray(inside), bins
    good, present, inside, bins = env.physical_cache[key]
    correct = np.asarray(rollout_metrics["correct_station_mask"], dtype=int)
    captured = np.asarray(rollout_metrics["capture_station_mask"], dtype=int)
    eligible_regions = np.unique(spec[4][good])
    outcomes = np.asarray(rollout_metrics["region_outcomes"])
    return {"physically_accessible_mm": float(bins[good].sum()), "raw_target_mm": float(bins[present].sum()),
            "target_inside_bone_mm": float(bins[present & inside].sum()), "physical_regions": len(eligible_regions),
            "correct_physical_mm": float(bins[correct[good[correct]]].sum()),
            "capture_physical_mm": float(bins[captured[good[captured]]].sum()),
            "correct_physical_regions": int(np.sum(outcomes[eligible_regions] == 1)),
            "strict_physical": int(len(eligible_regions) > 0 and np.all(outcomes[eligible_regions] == 1))}
