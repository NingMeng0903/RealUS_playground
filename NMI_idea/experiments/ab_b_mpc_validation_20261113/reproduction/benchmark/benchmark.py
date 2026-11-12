"""Run paired, simulated multi-visit scans on the frozen moving atlas.

The extension uses the original prefix, motion model, stress generator,
identity threshold and six planned methods/options. Test labels are consumed
by the evaluator after the public policy has completed its decisions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from search import choose


HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parent / "RealUS_playground"
CAPTURE_SLOT_S = .85
CAPTURE_LIMIT = 5
TRAVEL_CAP_MM = 60.
TURN_CAP_DEG = 60.
VIEWS = (3, 2, 4, 7, 8)
TASKS = {"daughter": ((3, 1), (4, 1)), "trunk": ((5, 0), (6, 0))}
ARMS = ("MPC", "MPC_rec1", "B_status", "B_bound")
SEEDS = {"smoke": 6173060, "development": 6173061, "confirmation": 6173062}
RESERVES = {"narrow": 10., "wide": 26.}
WORLD = MODEL = GRAPHS = PREFIX = API = NODES = None
PROCESS_CACHE = {}


@dataclass(frozen=True)
class Packet:
    reading: object
    site: int
    view: int
    centroids: np.ndarray


@dataclass
class Graph:
    targets: tuple
    points: tuple
    path_table: dict
    edges: np.ndarray
    bits: np.ndarray
    labels: tuple


def initialize(repo, samples, mode, preflight_cache):
    global WORLD, MODEL, GRAPHS, PREFIX, API, NODES
    if samples < 4 or samples & (samples - 1):
        raise ValueError("quadrature count must be a power of two")
    root = Path(repo) / "NMI_idea" / "experiments"
    y = root / "ab_tro_y_continuation_20261112"
    sys.path[:0] = [str(y), str(root / "ab_map_recheck_20261112")]
    from map_model import MapFilter, MapWorld
    import dynamic_scan
    from mechanics import MotionModel, Path as MotionPath
    from belief import (Reading, CONFIDENCE, HALF_WIDTH, DEPTH, IMAGE_SD,
                        NEAR_CLUTTER_RATE, NEAR_CLUTTER_SD, OUTLIER_RATE, OUTLIER_SD)

    with np.load(y / "review_inputs" / "POSED_BRANCH.npz", allow_pickle=False) as asset:
        posed = (asset["dense"], asset["s"], asset["daughter"],
                 tuple(map(int, asset["sites"])), int(asset["fork"]))
    with patch.object(dynamic_scan, "posed_branch", return_value=posed):
        WORLD = MapWorld(mode=mode)
    MODEL = MotionModel(WORLD)
    API = {
        "MapFilter": MapFilter, "Reading": Reading, "confidence": CONFIDENCE,
        "image_sd": IMAGE_SD, "near_rate": NEAR_CLUTTER_RATE,
        "near_sd": NEAR_CLUTTER_SD, "outlier_rate": OUTLIER_RATE, "outlier_sd": OUTLIER_SD,
        "lo": np.array([-HALF_WIDTH, DEPTH[0]]), "hi": np.array([HALF_WIDTH, DEPTH[1]]),
    }
    NODES = qmc.Sobol(16, scramble=True, seed=8211383).random_base2(samples.bit_length() - 1)
    center = WORLD.scan.central
    cache_path = Path(preflight_cache) if preflight_cache else None
    fingerprint = hashlib.sha256((y / 'mechanics.py').read_bytes()
                                 + WORLD.geometry_path.read_bytes() + repr((TASKS, VIEWS, CAPTURE_SLOT_S)).encode()).hexdigest()
    if cache_path and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as saved:
            meta = json.loads(str(saved['metadata']))
            if meta['fingerprint'] != fingerprint:
                raise ValueError('preflight cache is for a different geometry or protocol')
            for i, spec in enumerate(meta['paths']):
                spec['start'], spec['goal'] = tuple(spec['start']), tuple(spec['goal'])
                path = MotionPath(**spec)
                key = path.start, path.goal
                MODEL.path_meta[key] = path
                if path.legal:
                    MODEL.q_reference[key] = saved[str(i)]

    prefix_y = MODEL.register(WORLD.initial_route[-1], (3, center), 2.50)
    prefix_trunk = MODEL.register((3, center), (5, center), CAPTURE_SLOT_S)
    if not prefix_y.legal or not prefix_trunk.legal:
        raise RuntimeError("preflight failed for shared initial motion")
    PREFIX = {"daughter": (prefix_y,), "trunk": (prefix_y, prefix_trunk)}
    GRAPHS = {}
    if cache_path and cache_path.exists():
        graph_rows = map(register_graph, TASKS.items())
        for name, graph, _, _ in graph_rows:
            GRAPHS[name] = graph
    else:
        with ProcessPoolExecutor(max_workers=len(TASKS), mp_context=get_context('fork')) as pool:
            for name, graph, path_meta, references in pool.map(register_graph, TASKS.items()):
                GRAPHS[name] = graph
                MODEL.path_meta.update(path_meta)
                MODEL.q_reference.update(references)
        if cache_path:
            paths = list(MODEL.path_meta.values())
            metadata = json.dumps({'fingerprint': fingerprint, 'paths': [vars(path) for path in paths]})
            np.savez_compressed(cache_path, metadata=metadata, **{
                str(i): MODEL.q_reference[path.start, path.goal]
                for i, path in enumerate(paths) if path.legal})


def register_graph(item):
    name, targets = item
    points = tuple((site, view) for site, _ in targets for view in VIEWS if WORLD.allowed[site, view])
    site_bits = {site: 1 << i for i, (site, _) in enumerate(targets)}
    bits = np.array([site_bits[site] for site, _ in points])
    labels = tuple(dict(targets)[site] for site, _ in points)
    paths = {}
    edges = np.full((len(points), len(points), 3), np.inf)
    for i, start in enumerate(points):
        for j, goal in enumerate(points):
            path = MODEL.register(start, goal, CAPTURE_SLOT_S)
            if not path.legal:
                continue
            paths[i, j] = path
            matrices = MODEL.chart_reference(start, goal, np.array([0., 1.]), 15.)[:, :3, :3]
            a, b = Rotation.from_matrix(matrices)
            turn = float(np.rad2deg((a.inv() * b).magnitude()))
            edges[i, j] = (path.cost_mm, turn, path.stroke_mm)
    return name, Graph(targets, points, paths, edges, bits, labels), MODEL.path_meta, MODEL.q_reference


def public(frame):
    return Packet(API["Reading"](frame.t, frame.q, frame.velocity),
                  frame.site, frame.view, frame.centroids)


def confidence(filter_, packet, label):
    if not len(packet.centroids):
        return np.zeros(0)
    prior, _ = filter_.prior_at(packet.reading)
    likelihood, association = filter_.emission(
        WORLD, packet.reading, packet.site, packet.view, packet.centroids)
    joint = np.einsum("jwv,wv->j", association[:, label], prior)
    return joint / np.sum(likelihood * prior)


class Evidence:
    """Common public estimator; optional leave-one-view-out verification."""

    def __init__(self, estimator, graphs, block=False):
        self.filters = [estimator]
        self.blocks = tuple(view for _, view in graphs.points)
        self.blocks = tuple(dict.fromkeys(self.blocks)) if block else ()
        self.filters.extend(estimator.copy() for _ in self.blocks)

    def verify(self, packet, label):
        values = np.stack([confidence(f, packet, label) for f in self.filters])
        if not values.shape[1]:
            return None, 0.
        low = values.min(axis=0)
        ordinal = int(np.argmax(low))
        return (ordinal if low[ordinal] >= API["confidence"] else None, float(low[ordinal]))

    def update(self, packet, index):
        for deleted_view, estimator in zip((None, *self.blocks), self.filters):
            if deleted_view == packet.view:
                estimator.advance(packet.reading, track=False)
            else:
                estimator.observe(WORLD, packet.reading, packet.site, packet.view,
                                  packet.centroids, event=index, retain=False, track=False)

    def _forecast(self, reading, point, label):
        """Shared quadrature of accepted-correct and accepted-wrong masses."""
        filter_ = self.filters[0]
        site, view = point
        prior = np.stack([f.prior_at(reading)[0] for f in self.filters])
        base, rows = WORLD.projection(reading.q, site, view)
        count = len(base)
        means = filter_.means(base, rows, reading).reshape(-1, count, 2)
        cumulative = np.cumsum(prior[0].ravel())
        cumulative[-1] = 1.
        sample = means[np.searchsorted(cumulative, NODES[:, 0], side="right")]
        sd = np.where(NODES[:, 3:3 + count, None] < API["outlier_rate"],
                      API["outlier_sd"], API["image_sd"])
        values = sample + ndtri(NODES[:, 5:5 + 2 * count]).reshape(-1, count, 2) * sd
        lo, hi = API["lo"], API["hi"]
        inside = np.all((values >= lo) & (values <= hi), axis=2)
        keep = inside & (NODES[:, 1:1 + count] < WORLD.detection_probability(reading.q, site, view))
        mean = base.mean(axis=0)
        near_sd = API["near_sd"]
        lower, upper = ndtr((lo - mean) / near_sd), ndtr((hi - mean) / near_sd)
        near = mean + near_sd * ndtri(lower + NODES[:, 11:13] * (upper - lower))
        uniform = lo + NODES[:, 13:15] * (hi - lo)
        clutter = np.where(NODES[:, 10:11] < API["near_rate"], near, uniform)
        extra = NODES[:, 9] < WORLD.clutter_probability(site, view)
        total_correct, total_wrong = 0., 0.
        for coords, valid, has_clutter, artifact in zip(values, keep, extra, clutter):
            mask = coords[valid]
            if has_clutter:
                mask = np.vstack((mask, artifact))
            if not len(mask):
                continue
            likelihood, association = filter_.emission(WORLD, reading, site, view, mask)
            weights = prior.reshape(len(prior), -1)
            predicted = weights @ likelihood.ravel()
            joint = weights @ association[:, label].reshape(len(mask), -1).T
            probabilities = joint / predicted[:, None]
            robust = probabilities.min(axis=0)
            selected = int(np.argmax(robust))
            if robust[selected] >= API["confidence"]:
                probability = probabilities[0, selected]
                total_correct += probability
                total_wrong += 1. - probability
        return total_correct / len(NODES), total_wrong / len(NODES)

    def forecast(self, slots, graph):
        base = self.filters[0]
        readings = tuple(base.predicted_reading(base.reading.t + CAPTURE_SLOT_S * j)
                         for j in range(1, slots + 1))
        return np.asarray([[self._forecast(r, point, label)
                            for point, label in zip(graph.points, graph.labels)]
                           for r in readings])


def move(case, path, clock, q, cache):
    key = (path.id, clock, np.asarray(q).tobytes())
    if key not in cache:
        trace, exit_q = MODEL.execute(path, clock, lambda t: case.true_angle_rate(t)[0], q)
        costs = (trace["travel_mm"], trace["turn_deg"], trace["duration_s"], trace["stroke_mm"])
        frame = WORLD.observe(case, trace["t_end"], *path.goal) if trace["legal_completed"] else None
        cache[key] = frame, costs, exit_q, trace["t_end"]
    return cache[key]


def start_task(case, task, cache):
    frames = WORLD.initial(case)
    filter_ = API["MapFilter"]()
    for idx, frame in enumerate(frames):
        packet = public(frame)
        filter_.observe(WORLD, packet.reading, packet.site, packet.view, packet.centroids,
                        event=idx, retain=False, track=False)
    clock = frames[-1].t
    q, _ = MODEL.command(PREFIX[task][0], 0., case.true_angle_rate(clock)[0])
    for idx, path in enumerate(PREFIX[task], start=len(frames)):
        frame, _, q, clock = move(case, path, clock, q, cache)
        if frame is None:
            return None
        packet = public(frame)
        filter_.observe(WORLD, packet.reading, packet.site, packet.view, packet.centroids,
                        event=idx, retain=False, track=False)
    return filter_, q, clock


def run_policy(case, task, reserve, method, estimator_mode, start, cache):
    evidence = Evidence(start[0].copy(), GRAPHS[task], block=estimator_mode == "block")
    q, clock = start[1:]
    graph = GRAPHS[task]
    at = graph.points.index(PREFIX[task][-1].goal)
    pending = 3
    costs = np.zeros(4)
    confirmed = []
    steps = []
    planning_s = 0.
    reason = "capture_limit"
    for tick in range(CAPTURE_LIMIT):
        before = time.perf_counter()
        # The computation budget counts predictions, including reused cached
        # ones. Wall clock is separately logged as an offline runtime measure.
        key = (task, estimator_mode, tick, evidence.filters[0].reading,
               tuple(f.weights.tobytes() for f in evidence.filters))
        if key not in PROCESS_CACHE:
            PROCESS_CACHE[key] = evidence.forecast(CAPTURE_LIMIT - tick, graph)
        scores = PROCESS_CACHE[key]
        budget = np.array([TRAVEL_CAP_MM - costs[0], TURN_CAP_DEG - costs[1], reserve - costs[3]])
        value, action, nodes = choose(scores, graph.edges, graph.bits, at, budget,
                                      pending, method)
        planning_s += time.perf_counter() - before
        if action < 0:
            reason = "stop_without_completion"
            break
        path = graph.path_table[at, action]
        frame, used, q, clock = move(case, path, clock, q, cache)
        costs += used
        if frame is None:
            reason = "motion_failure"
            break
        if (costs[0] > TRAVEL_CAP_MM + 1.e-8 or costs[1] > TURN_CAP_DEG + 1.e-8
                or costs[3] > reserve + 1.e-8):
            reason = "resource_failure"
            break
        packet = public(frame)
        bit = int(graph.bits[action])
        ordinal, verification = None, 0.
        if pending & bit:
            ordinal, verification = evidence.verify(packet, graph.labels[action])
            if ordinal is not None:
                pending ^= bit
                confirmed.append((tick, ordinal, graph.labels[action], frame))
        evidence.update(packet, 64 + tick)
        steps.append({"action": list(graph.points[action]), "from": list(graph.points[at]),
                      "pending_after": pending, "declared": ordinal, "confidence": verification,
                      "cost": list(used), "value": value, "search_nodes": nodes,
                      "forecast": scores[0].tolist(), "time": packet.reading.t,
                      "q": packet.reading.q, "velocity": packet.reading.velocity,
                      "centroids_mm": packet.centroids.tolist()})
        at = action
        if pending == 0:
            reason = "declared_complete"
            break
    labels = [frame.labels[ordinal] == target for _, ordinal, target, frame in confirmed]
    wrong = not all(labels)
    return {"success": pending == 0 and not wrong, "wrong": wrong, "reason": reason,
            "costs": costs.tolist(), "confirmed_eval_only": labels,
            "steps": steps, "planning_s": planning_s}


def paired(job):
    index, split, tasks, reserves, arms, modes = job
    PROCESS_CACHE.clear()
    seed = int(np.random.SeedSequence([SEEDS[split], index]).generate_state(1)[0])
    case = WORLD.case(seed)
    cache = {}
    records = []
    for task in tasks:
        start = start_task(case, task, cache)
        for reserve_name in reserves:
            for mode in modes:
                for arm in arms:
                    row = (run_policy(case, task, RESERVES[reserve_name], arm, mode, start, cache)
                           if start is not None else
                           {"success": False, "wrong": False, "reason": "shared_prefix_motion_failure",
                            "costs": [0., 0., 0., 0.], "confirmed_eval_only": [],
                            "steps": [], "planning_s": 0.})
                    records.append(dict(row, index=index, seed=seed, task=task, reserve=reserve_name,
                                        arm=arm, estimator=mode))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=SEEDS, default="smoke")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--preflight-cache", type=Path)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--mode", choices=("stress", "nominal"), default="stress")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--reserves", nargs="+", choices=RESERVES, default=list(RESERVES))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--estimators", nargs="+", choices=("nominal", "block"), default=["nominal"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    initialize(args.repo, args.samples, args.mode, args.preflight_cache)
    settings = vars(args) | {
        "preflight_s": time.perf_counter() - start,
        "seed_base": SEEDS[args.split], "captures": CAPTURE_LIMIT, "slot_s": CAPTURE_SLOT_S,
        "travel_cap_mm": TRAVEL_CAP_MM, "turn_cap_deg": TURN_CAP_DEG, "stroke_mm": RESERVES,
        "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                          for name in ("benchmark.py", "search.py")},
        "graphs": {task: {"points": GRAPHS[task].points,
                          "targets": GRAPHS[task].targets,
                          "paths": [vars(path) for path in GRAPHS[task].path_table.values()]}
                   for task in args.tasks},
    }
    (args.out / "settings.json").write_text(json.dumps(settings, default=str, indent=2) + "\n")
    # Initialize Numba before fork. All methods see an identical preflight.
    dummy = np.ones((2, 2, 2)) * .2
    for method in ARMS:
        choose(dummy, np.zeros((2, 2, 3)), np.array([1, 2]), 0, np.full(3, 10.), 3, method)
    jobs = [(idx, args.split, args.tasks, args.reserves, args.arms, args.estimators)
            for idx in range(args.start, args.start + args.n)]
    counts = Counter()
    totals = Counter()
    print(f"Preflight ready in {settings['preflight_s']:.1f}s", flush=True)
    with (args.out / "cases.jsonl").open("w") as out:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("fork")) as pool:
            for idx, rows in enumerate(pool.map(paired, jobs), 1):
                for row in rows:
                    out.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                    group = "/".join(row[key] for key in ("task", "reserve", "estimator", "arm"))
                    counts[group] += row["success"]
                    totals[group] += 1
                out.flush()
                if idx == 1 or idx % 8 == 0 or idx == len(jobs):
                    print(f"{idx}/{len(jobs)} paired cases, {time.perf_counter() - start:.1f}s",
                          flush=True)
    summary = {group: {"correct": counts[group], "n": total} for group, total in totals.items()}
    (args.out / "counts.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
