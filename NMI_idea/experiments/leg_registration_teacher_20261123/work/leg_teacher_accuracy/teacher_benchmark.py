"""Evaluate privileged scan teachers on the named v2 routes and fixed sensor tapes."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import lru_cache
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np
from scipy.special import ndtr

DEFAULT_HANDOFF = Path(__file__).resolve().parents[2] / "outputs" / "leg_scan_cli_handoff"
HANDOFF = Path(os.environ.get("LEG_SCAN_HANDOFF", DEFAULT_HANDOFF))
LEGACY = HANDOFF / "leg_closed_loop_20261122_v1"
V2 = HANDOFF / "leg_scan_target_routes_20261123_v2"
sys.path.insert(0, str(LEGACY / "src"))
os.environ["LEG_ENABLE_DIRD"] = "0"
os.environ.setdefault("LEG_ENABLE_MATERIAL_ATLAS", "1")

import motion
import run_online as run
from environment import ScanEnvironment, Score, spec_for
from measurement_evidence import synthetic_type_signal, target_marginal
from online_reader import PublicHistoryReader
from scan_kernel import DEPTH_MAX, DEPTH_MIN, LATERAL, Trial, association
from synthetic_anatomy import LayoutFactory

TASKS = {"femoral": (0, 2), "groin_to_distal_calf": (2,)}
PROFILE_NAMES = ("ordinary", "stress")


class GeometryForecast:
    """Current anatomy and expected sensor quality, with no future noise draws."""

    def __init__(self, env, trial):
        self.env, self.trial = env, trial
        self.stress = trial.stress

    def blur(self, station, sector):
        window = self.trial.window
        if window is None:
            return 0.
        ds = (station - window.station) / 1.15
        da = motion._sector_delta(window.sector, sector) / .80
        return float(np.exp(-.5 * (ds * ds + da * da)))

    @lru_cache(maxsize=24000)
    def view(self, station, sector):
        scanner, trial = self.env.scanner, self.trial
        points, gate = scanner._anatomy.training_truth(scanner.surface, trial, station, sector)
        x, depth = points.T
        inside = (np.abs(x) <= LATERAL) & (depth >= DEPTH_MIN) & (depth <= DEPTH_MAX)
        if not inside[0]:
            return 0., 0., False
        coupling = 1. - .12 * abs(np.sin(sector * motion.ANGLE_STEP))
        detection = (.69 if self.stress else .91) * (.82 + .18 * coupling) * (1. - .96 * gate)
        margin = min(LATERAL - abs(x[0]), depth[0] - DEPTH_MIN, DEPTH_MAX - depth[0])
        fov_reliability = .55 + .45 * min(1., margin / 4.)
        competitors = np.flatnonzero(inside[1:]) + 1
        distance2 = np.square(points[competitors] - points[0]).sum(axis=1)
        blur = self.blur(station, sector)
        dx_dz = (points[competitors] - points[0]) / [4.7 + 10. * blur, 4.7 + 7. * blur]
        proximity = np.exp(-.5 * np.square(dx_dz).sum(axis=1))
        merge_rate = (.79 if self.stress else .44) * (1. - blur) + .99 * blur
        merge = merge_rate * np.max(proximity * detection[competitors], initial=0.)
        capture = detection[0] * fov_reliability * (1. - merge)
        separation = 1. - np.max(np.exp(-.5 * distance2 / 6.**2) * detection[competitors], initial=0.)
        return float(capture), float(capture * (.44 + .56 * separation)), True

    def __call__(self, station, sector):
        return self.view(round(float(station) * 4.) / 4., round(float(motion.wrap_sector(sector)) * 4.) / 4. % 24.)

    def posterior(self, frame):
        station, sector, echoes, quality = frame[0], frame[1], frame[2], frame[4]
        if not len(echoes):
            return np.empty(0)
        scanner = self.env.scanner
        points, gate = scanner._anatomy.training_truth(scanner.surface, self.trial, station, sector)
        present = np.isfinite(points).all(axis=1)
        points = np.nan_to_num(points, nan=999.)
        expected = (.68 if self.stress else .89) * (1. - .96 * gate) * present
        sigma = [2.35, 1.85]
        high = ndtr((np.array([LATERAL, DEPTH_MAX]) - points) / sigma)
        low = ndtr((np.array([-LATERAL, DEPTH_MIN]) - points) / sigma)
        integration = (high - low).prod(axis=-1)
        _, probabilities, _, _ = association(points[None], expected[None], integration[None],
                                             echoes, quality, tuple(range(len(points))))
        return probabilities[0]

    def joint_marginal(self, frame, type_signal):
        station, sector, echoes = frame[:3]
        if not len(echoes):
            return np.empty(0)
        scanner = self.env.scanner
        points, gate = scanner._anatomy.training_truth(scanner.surface, self.trial, station, sector)
        x, depth = points.T
        inside = (np.abs(x) <= LATERAL) & (depth >= DEPTH_MIN) & (depth <= DEPTH_MAX)
        if not inside[0]:
            return np.zeros(len(echoes))
        safe = np.nan_to_num(points, nan=999.)
        coupling = 1. - .12 * abs(np.sin(sector * motion.ANGLE_STEP))
        detection = (.69 if self.stress else .91) * (.82 + .18 * coupling) * (1. - .96 * gate)
        sigma = [2.3, 1.8] if self.stress else [1.4, 1.1]
        high = ndtr((np.array([LATERAL, DEPTH_MAX]) - safe) / sigma)
        low = ndtr((np.array([-LATERAL, DEPTH_MIN]) - safe) / sigma)
        retention = (high - low).prod(axis=-1)
        detection *= retention
        return target_marginal(safe, detection, inside, echoes, self.stress, self.blur(station, sector), type_signal)


@lru_cache(maxsize=6)
def load_environment(pack_path, shape):
    pack = Path(pack_path)
    data = motion._load_pack(pack)
    centers = data["centers"][shape]
    origins, radii, y = (data[key][shape] for key in ("origins", "radii", "ys"))
    contacts = origins[:, None, :] + radii[..., None] * motion.legacy_geometry.NORMALS[None]
    chart = (origins, radii, data["origins"][0], data["radii"][0], data["ys"][0])
    surface = motion.Surface(y, contacts, centers.copy(), contacts[..., 0] < -1., shape=chart)
    links = data["links"]
    surface.reference_links = links
    scanner = motion.Scanner(data["neutral_grid"], surface, motion.Anatomy(centers, links=links),
                             edge_status=data["edge_status"])
    return ScanEnvironment(scanner, LayoutFactory(pack, scanner, shape))


def select_cases(manifest, split, cases_per_route):
    seed = {"diagnostic": run.DEV_SEED, "verification": 395_780_013, "verification_2": 284_611_973}[split]
    quotas = {task: cases_per_route // (len(sources) * len(PROFILE_NAMES)) for task, sources in TASKS.items()}
    sizes = Counter()
    cases = []
    for case in run.run_cases(cases_per_route * 24, "development", seed):
        if case.task not in quotas or case.source not in manifest["routes"][case.task]["eligible_source_indices"]:
            continue
        group = case.task, case.source, case.profile
        if sizes[group] >= quotas[case.task]:
            continue
        sizes[group] += 1
        cases.append(replace(case, challenge=False))
        if len(cases) == cases_per_route * len(TASKS):
            break
    if len(cases) != cases_per_route * len(TASKS):
        raise RuntimeError("Insufficient balanced eligible cases for both named routes")
    return cases


def geometry_audit(env, spec, initial, forecast):
    route = env.route
    lo, hi = spec[1:3]
    reachable = np.zeros((hi - lo + 1, motion.SECTORS), dtype=bool)
    reachable[0, int(initial.sector)] = True
    for i, current in enumerate(reachable[:-1]):
        reachable[i + 1, route.sectors[current[:, None] & route.legal[lo + i]]] = True
    backward, _ = route.reserve(lo, hi)
    reachable &= backward < 100_000
    geometry = []
    for region in range(6):
        supports = np.flatnonzero(spec[4] == region)
        values, connected_values = [], []
        target_count, open_count = 0, 0
        for k in supports:
            station = k + lo
            for sector in np.flatnonzero(env.scanner.valid[station]):
                cv, qv, visible = forecast(station, sector)
                target_count += int(np.isfinite(forecast.trial.layout.centers[station, 0]).all())
                open_count += int(visible)
                if visible:
                    values.append((cv, qv, station, int(sector)))
                    if reachable[k, sector]:
                        connected_values.append((cv, qv, station, int(sector)))
        geometry.append({"region": region, "axial_bounds_mm": [float(env.scanner.stations[lo + supports[0]]),
                                                               float(env.scanner.stations[lo + supports[-1]])],
                         "valid_poses_with_target": target_count, "in_frame_poses": open_count,
                         "best": max(values, default=None), "best_from_start": max(connected_values, default=None)})
    return geometry


def coverage_metrics(env, spec, node, threshold):
    score = Score(spec, env.scanner.surface.y)
    score.update(node.reader)
    rec = score.delivery(node.reader, threshold, node.station == spec[2])
    by_region = []
    for region in range(6):
        frames = [frame for frame in node.reader.frames if frame[5] == region]
        true_count = sum(bool(0 in frame[3]) for frame in frames)
        by_region.append({"frames": len(frames), "true_frames": true_count,
                          "merges": sum(bool(-2 in frame[3]) for frame in frames),
                          "false_echo": sum(bool(-1 in frame[3]) for frame in frames)})
    return rec | {"frames": len(node.reader.frames), "ticks": node.tick, "distance_mm": float(node.distance),
                  "by_region": by_region}


def selection_metrics(frames, selections):
    outcomes = [-1 if choice is None else int(frames[choice[0]][3][choice[1]] == 0)
                for choice in selections]
    return {"pass": int(outcomes == [1] * 6), "region_outcomes": outcomes,
            "correct_regions": outcomes.count(1), "wrong_regions": outcomes.count(0),
            "wrong_delivery": int(0 in outcomes),
            "scores": [None if choice is None else choice[2] for choice in selections]}


def type_signal(frame, tape_seed):
    rng = np.random.default_rng(np.random.SeedSequence([tape_seed, frame[8], 2107]))
    return synthetic_type_signal(frame[3], rng)


def teacher_selections(frames, score_frame, threshold):
    """Rank anonymous echoes with the privileged geometry of the current layout."""
    best = [None] * 6
    for frame_id, frame in enumerate(frames):
        probabilities = score_frame(frame)
        if not len(probabilities):
            continue
        slot = int(np.argmax(probabilities))
        score = float(probabilities[slot])
        region = frame[5]
        if score >= threshold and (best[region] is None or score > best[region][2]):
            best[region] = (frame_id, slot, score)
    return best


@lru_cache(maxsize=16)
def supported_model(scanner, allowed_sources):
    """Condition the public prior on which reference atlases cover the task."""
    model = motion.Model(scanner.surface, 27.)
    hypotheses = model.prior.reshape(3, -1)
    unavailable = [index for index in range(3) if index not in allowed_sources]
    hypotheses[unavailable] = 0.
    model.prior /= model.prior.sum()
    return model


def route_conditioned_selections(node, spec, model, threshold):
    frames = node.reader.frames
    reader = PublicHistoryReader(model, spec[3], True)
    for frame in frames:
        station, sector, echoes, quality, region, tick = frame[0], frame[1], frame[2], frame[4], frame[5], frame[8]
        reader.push(station, sector, echoes, np.full(len(echoes), -9), quality, region, tick)
    return [reader.select(region, threshold) for region in range(6)]


def route_path(env, spec, station, sector, slot):
    route = env.route
    key = station, sector, int(slot)
    path = route.cache.get(key)
    if path is None:
        ticks = route.clock[station, sector, slot]
        fraction = np.arange(ticks + 1) / ticks
        positions = np.column_stack((station + fraction, sector + route.deltas[slot] * fraction))
        lengths = env.scanner._path_metrics(positions, spec)
        if lengths is None:
            raise RuntimeError("Teacher chose a move outside the common contact controller")
        path = route.cache[key] = (positions[1:], lengths)
    return path


def apply_route_slot(env, spec, node, trial, limits, slot):
    """Execute one of the common route controller's budgeted, legal moves."""
    route = env.route
    station, sector = int(node.station), int(node.sector)
    if not route.bounded_candidates(node, spec, limits)[slot]:
        raise RuntimeError("Teacher chose a budget-infeasible route move")
    path = route_path(env, spec, station, sector, slot)
    for (s, a), length in zip(*path):
        node.tick += 1
        node.distance += float(length)
        env.scanner._observe(node, spec, trial, float(s), float(a), node.tick)
    node.station, node.sector = station + 1, int(route.sectors[sector, slot])



class PrivilegedTeacher:
    def __init__(self, env, spec, trial, mode, task, threshold):
        self.env, self.spec, self.trial = env, spec, trial
        self.forecast = GeometryForecast(env, trial)
        self.mode = mode
        self.dwell_ticks = 0
        self.probes = []
        self.forward_changes = 0
        self.confirmed = np.zeros(6, dtype=bool)
        self.seen_frames = 0
        self.adaptive = mode in ("adaptive", "adaptive_type", "search", "search_type")
        self.use_type_signal = mode in ("adaptive_type", "search_type")
        self.search_forward = mode in ("search", "search_type") and task == "groin_to_distal_calf"
        self.threshold = threshold
        self.planning = []
        if self.adaptive:
            for fork in (1, 2, 3):
                # Simulated candidate suffixes share three teacher-only tapes.
                # The rollout tape is used only when executing a selected plan.
                seed = int(np.random.SeedSequence([trial.world_seed, trial.tape_seed, fork, 4257])
                           .generate_state(1, dtype=np.uint64)[0])
                planner = Trial(trial.source, seed, trial.profile, world_seed=trial.world_seed, split=trial.split)
                planner.layout, planner.window = trial.layout, trial.window
                self.planning.append((planner, GeometryForecast(env, planner)))
        steer = mode in ("route", "verify") or (self.adaptive and task == "femoral")
        self.move_values = self.route_values() if steer else None

    def route_values(self):
        """Value of each legal move plus the best continuation to the terminal anchor."""
        route, spec = self.env.route, self.spec
        first, last = spec[1:3]
        values = np.full((last - first, motion.SECTORS, len(route.deltas)), -np.inf)
        tail = np.where(self.env.scanner.valid[last], 0., -np.inf)
        for station in range(last - 1, first - 1, -1):
            i = station - first
            for sector in np.flatnonzero(self.env.scanner.valid[station]):
                for slot in np.flatnonzero(route.legal[station, sector]):
                    next_sector = route.sectors[sector, slot]
                    if not np.isfinite(tail[next_sector]):
                        continue
                    mid = self.forecast(station + .5, sector + .5 * route.deltas[slot])[1]
                    destination = self.forecast(station + 1, next_sector)[1]
                    value = .35 * mid + .65 * destination
                    values[i, sector, slot] = value - .008 * route.distance[station, sector, slot] + tail[next_sector]
            tail = values[i].max(axis=1)
        return values

    def update_confirmation(self, node):
        for frame in node.reader.frames[self.seen_frames:]:
            region = frame[5]
            if self.confirmed[region]:
                continue
            if self.adaptive:
                cue = type_signal(frame, self.trial.tape_seed) if self.use_type_signal else np.empty(0)
                posterior = self.forecast.joint_marginal(frame, cue)
                threshold = self.threshold
            else:
                posterior = self.forecast.posterior(frame)
                threshold = .94
            self.confirmed[region] = posterior.max(initial=0.) >= threshold
        self.seen_frames = len(node.reader.frames)

    def compare_plans(self, node, plans, region):
        """Score correct and wrong commitments on independent candidate suffixes."""
        utilities = []
        for poses in plans:
            totals = []
            for trial, forecast in self.planning:
                best_score, chosen_is_target = self.threshold, None
                for index, (station, sector) in enumerate(poses, start=node.tick + 1):
                    if motion.region_of(self.spec, station) != region:
                        continue
                    sensor = self.env.scanner
                    points, gate = sensor._anatomy.training_truth(sensor.surface, trial, station, sector)
                    coupling = 1. - .12 * abs(np.sin(sector * motion.ANGLE_STEP))
                    echoes, labels, quality = trial.observe(points, gate, index, station=station, sector=sector,
                                                           contact_quality=coupling)
                    frame = (station, sector, echoes, labels, quality, region, None, None, index)
                    cue = type_signal(frame, trial.tape_seed) if self.use_type_signal else np.empty(0)
                    posterior = forecast.joint_marginal(frame, cue)
                    if len(posterior):
                        slot = int(np.argmax(posterior))
                        if posterior[slot] > best_score:
                            best_score = float(posterior[slot])
                            chosen_is_target = bool(labels[slot] == 0)
                totals.append(1. if chosen_is_target is True else -.75 if chosen_is_target is False else 0.)
            utilities.append(float(np.mean(totals)) - .004 * len(poses))
        return utilities

    def maybe_probe(self, node, limits, region, close_to_exit):
        if not close_to_exit or self.confirmed[region] or len(self.probes) >= 12:
            return node
        spec = self.spec
        station, sector = int(node.station), int(node.sector)
        caps = limits.probe(node, spec)
        if caps[0] < 6 or caps[1] < 5.:
            return node
        goals = []
        for skill in (1, 2):
            if skill == 2 and station == spec[1]:
                continue
            for turn in (-.86, -.44, .0, .44, .86):
                if skill == 1 and turn == 0.:
                    continue
                for axial in ((0., .8) if skill == 1 else (.5, .95)):
                    controls = np.array([turn, axial, 0.])
                    goal = self.env.material._skill_goal(np.array([station, sector], dtype=float), controls, skill, spec)
                    if not self.env.scanner.surface.legal(*goal) or motion.region_of(spec, goal[0]) != region:
                        continue
                    p = self.forecast(*goal)[1]
                    if p >= .14:
                        goals.append((p, skill, controls))
        best = None
        sampled_plans = []
        current = self.forecast(station, sector)[1] if self.mode == "verify" else 0.
        for _, skill, controls in sorted(goals, key=lambda item: item[0], reverse=True)[:5]:
            plan = self.env.material.plan_probe(node, spec, skill, controls, *caps, use_dird=False)
            if plan is None:
                continue
            if self.adaptive:
                sampled_plans.append(plan)
                continue
            views = [self.forecast(*point)[1] for segment in plan.segments for point in segment[0][1:]
                     if motion.region_of(spec, point[0]) == region]
            value = max(views, default=0.) + .18 * min(1., sum(views) / 2.) - .006 * plan.ticks
            if value > current + .07 and (best is None or value > best[0]):
                best = value, plan
        if sampled_plans:
            paths = [np.concatenate([segment[0][1:] for segment in plan.segments]) for plan in sampled_plans]
            scores = self.compare_plans(node, paths, region)
            top = int(np.argmax(scores))
            if scores[top] > 0.:
                best = scores[top], sampled_plans[top]
        if best is None:
            return node

        def stop_after_confirmation(observed, _info):
            self.update_confirmation(observed)
            return self.confirmed[region]

        before = node.tick
        result = self.env.material.run_probe(node, spec, self.trial, best[1], *caps, feedback=True,
                                             stop=stop_after_confirmation, use_dird=False, copy_node=False)
        if not result.legal or result.node.station != station or result.node.sector != sector:
            raise RuntimeError("Teacher probe failed to rejoin the route")
        self.probes.append({"region": region, "station": station, "ticks": result.node.tick - before})
        self.update_confirmation(result.node)
        return result.node

    def advance(self, node, limits, index, region, end_of_region):
        env, spec = self.env, self.spec
        station, sector = int(node.station), int(node.sector)
        if self.move_values is not None:
            legal = env.route.bounded_candidates(node, spec, limits)
            slot = int(np.argmax(np.where(legal, self.move_values[index, sector], -np.inf)))
            apply_route_slot(env, spec, node, self.trial, limits, slot)
        elif (self.search_forward and not self.confirmed[region]
              and 1 <= end_of_region - station <= 10 and (end_of_region - station) % 2 == 1):
            continuation = node.fork()
            env.route.advance(continuation, spec, self.planning[0][0], limits)
            reference = int(np.flatnonzero(env.route.sectors[sector] == continuation.sector)[0])
            candidates = np.flatnonzero(env.route.bounded_candidates(node, spec, limits))
            paths = [route_path(env, spec, station, sector, slot)[0] for slot in candidates]
            scores = dict(zip(candidates, self.compare_plans(node, paths, region)))
            best = max(scores, key=scores.get)
            chosen = best if scores[best] > scores[reference] + .18 else reference
            self.forward_changes += int(chosen != reference)
            apply_route_slot(env, spec, node, self.trial, limits, chosen)
        else:
            env.route.advance(node, spec, self.trial, limits)

    def run(self, node, limits):
        spec, env = self.spec, self.env
        while True:
            self.update_confirmation(node)
            station, sector = int(node.station), int(node.sector)
            index = station - spec[1]
            region = int(spec[4][index])
            end_of_region = spec[1] + int(np.searchsorted(spec[4], region + 1)) - 1
            close_to_exit = end_of_region - station <= 2
            if (self.mode == "verify" or self.adaptive) and close_to_exit and not self.confirmed[region]:
                # Acquire extra independent innovations at an already legal pose.
                if self.forecast(station, sector)[0] >= .12:
                    ticks, _ = limits.probe(node, spec)
                    count = min(6 if self.adaptive else 5, ticks)
                    if self.adaptive:
                        pose = np.repeat([[station, sector]], count, axis=0)
                        if self.compare_plans(node, [pose], region)[0] <= 0.:
                            count = 0
                    for _ in range(count):
                        node.tick += 1
                        self.dwell_ticks += 1
                        env.scanner._observe(node, spec, self.trial, float(station), float(sector), node.tick)
                        self.update_confirmation(node)
                        if self.confirmed[region]:
                            break
            if self.mode in ("probe", "verify") or self.adaptive:
                node = self.maybe_probe(node, limits, region, close_to_exit)
            if station == spec[2]:
                break
            self.advance(node, limits, index, region, end_of_region)
        if node.tick > limits.ticks or node.distance > limits.mm + 1e-6:
            raise RuntimeError("Teacher exceeded the common end-to-end budget")
        return node


def scan_audit(env, case, threshold, need_geometry, mode):
    spec = spec_for(env.scanner, case)
    node, trial, limits = env.start(case, spec)
    forecast = GeometryForecast(env, trial)
    geometry = geometry_audit(env, spec, node, forecast) if need_geometry else None
    teacher = None
    if mode == "mpc":
        while node.station < spec[2]:
            env.route.advance(node, spec, trial, limits)
    else:
        teacher = PrivilegedTeacher(env, spec, trial, mode, case.task, threshold)
        node = teacher.run(node, limits)
    result = coverage_metrics(env, spec, node, threshold)
    frames = node.reader.frames
    exact = teacher_selections(frames, forecast.posterior, threshold)
    model = supported_model(env.scanner, TASKS[case.task])
    conditioned = route_conditioned_selections(node, spec, model, threshold)
    unknown_type = np.empty(0)
    joint_score = lambda frame: forecast.joint_marginal(frame, unknown_type)
    joint = teacher_selections(frames, joint_score, threshold)
    typed = teacher_selections(frames, lambda frame: forecast.joint_marginal(
        frame, type_signal(frame, case.tape_seed)), threshold)
    result.update({"geometry_reader": selection_metrics(frames, exact),
                   "supported_prior_reader": selection_metrics(frames, conditioned),
                   "joint_reader": selection_metrics(frames, joint),
                   "type_signal_reader": selection_metrics(frames, typed),
                   "dwell_ticks": 0 if teacher is None else teacher.dwell_ticks,
                   "forward_changes": 0 if teacher is None else teacher.forward_changes,
                   "probes": [] if teacher is None else teacher.probes,
                   "limit_ticks": limits.ticks, "limit_mm": limits.mm,
                   "layout_id": trial.layout.identifier})
    return geometry, result


def run_case(job):
    case, pack, threshold, tapes, arms = job
    env = load_environment(pack, case.shape)
    row = {"case": case.__dict__, "tapes": []}
    for number in range(tapes):
        sensor_case = replace(case, tape_seed=case.tape_seed + number * 1_002_003_007)
        outcomes = {}
        for index, arm in enumerate(arms):
            geometry, outcomes[arm] = scan_audit(env, sensor_case, threshold, number == 0 and index == 0, arm)
            if geometry is not None:
                row["geometry"] = geometry
        if len({report["layout_id"] for report in outcomes.values()}) != 1:
            raise RuntimeError("Matched teacher arms drew different anatomy")
        row["tapes"].append({"tape_seed": int(sensor_case.tape_seed)} | outcomes)
    return row


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--v2", default=V2, type=Path)
    parser.add_argument("--split", choices=["diagnostic", "verification", "verification_2"], default="diagnostic")
    parser.add_argument("--cases-per-route", type=int, default=12)
    parser.add_argument("--tapes", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--arms", nargs="+", choices=["mpc", "probe", "route", "verify", "adaptive",
                                                     "adaptive_type", "search", "search_type"], default=["mpc"])
    args = parser.parse_args()
    if args.cases_per_route < 4 or args.cases_per_route % 4:
        parser.error("--cases-per-route must be a positive multiple of four")
    split_manifest = json.loads((args.v2 / "atlas_routes_manifest.json").read_text())
    pack = args.v2 / "atlas_routes.npz"
    if run.digest(pack) != split_manifest["pack_sha256"]:
        raise RuntimeError("Atlas projection inputs changed")
    threshold = json.loads((LEGACY / "smoke/verification.json").read_text())["threshold"]
    cases = select_cases(split_manifest, args.split, args.cases_per_route)
    jobs = [(case, str(pack), threshold, args.tapes, args.arms) for case in cases]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=run.protect_holdout) as pool:
        rows = list(pool.map(run_case, jobs, chunksize=1))
    print(json.dumps({"atlas_sha256": split_manifest["pack_sha256"], "split": args.split, "threshold": threshold,
                      "tapes_per_case": args.tapes, "arms": args.arms, "cases": rows},
                     default=lambda scalar: scalar.item(), indent=2))


if __name__ == "__main__":
    main()
