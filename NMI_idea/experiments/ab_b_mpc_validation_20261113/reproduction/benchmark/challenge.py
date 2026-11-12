"""Test acquisition at a reference station before fresh scan completion.

The same unlabeled masks, estimator, motion graph and real-time tail selector
are used within each comparison. The target and reference beliefs share the
motion/observation likelihood, but the reference belief omits nearby target
masks to test dependence on that possibly correlated evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from benchmark import DEFAULT_REPO

TASKS = {"daughter": (5, 4, 3, 1), "trunk": (5, 6, 7, 0)}
SEED_BASES = {"smoke": 6173090, "development": 6173091, "confirmation": 6173092}
ARMS = ("MPC", "B_feedback", "B_challenge", "Ref_fixed", "Near_fixed")
MODES = ("nominal", "fitted", "reference")
FIRST_VIEWS = (3, 7, 8)
TAIL_VIEWS = (3, 2, 4, 7, 8)
SLOT = .85
MAX_TRAVEL_MM, MAX_TURN_DEG, MAX_STROKE_MM = 60., 60., 26.
FIRST_MOVE_PRICE, TIME_PRICE = .00045, .055
MODELS = {}


@dataclass(frozen=True)
class Packet:
    reading: object
    goal: tuple[int, int]
    mask: np.ndarray


@dataclass
class Graph:
    prefix: tuple
    first: tuple
    tails: dict


@dataclass
class Kernel:
    masks: tuple
    likelihood: np.ndarray
    association: np.ndarray
    proposal: np.ndarray


def packet(frame):
    return Packet(Reading(frame.t, frame.q, frame.velocity), (frame.site, frame.view),
                  np.array(frame.centroids, copy=True))


def nominal_samples(belief, reading, goal, draws):
    site, view = goal
    prior, _ = belief.prior_at(reading)
    base, design = WORLD.projection(reading.q, site, view)
    count = len(base)
    centers = belief.means(base, design, reading).reshape(-1, count, 2)
    cdf = np.cumsum(prior.ravel())
    cdf[-1] = 1.
    expected = centers[np.searchsorted(cdf, draws[:, 0], side="right")]
    sd = np.where(draws[:, 3:3 + count, None] < OUTLIER_RATE, OUTLIER_SD, IMAGE_SD)
    values = expected + ndtri(draws[:, 5:5 + 2 * count]).reshape(-1, count, 2) * sd
    inside = np.all((values >= LOWER) & (values <= UPPER), axis=2)
    valid = inside & (draws[:, 1:1 + count] < WORLD.detection_probability(reading.q, *goal))
    mean = base.mean(axis=0)
    lo, hi = ndtr((LOWER - mean) / NEAR_CLUTTER_SD), ndtr((UPPER - mean) / NEAR_CLUTTER_SD)
    near = mean + NEAR_CLUTTER_SD * ndtri(lo + draws[:, 11:13] * (hi - lo))
    uniform = LOWER + draws[:, 13:15] * (UPPER - LOWER)
    clutter = np.where(draws[:, 10:11] < NEAR_CLUTTER_RATE, near, uniform)
    added = draws[:, 9] < WORLD.clutter_probability(*goal)
    result = []
    for points, keep, false, extra in zip(values, valid, clutter, added):
        mask = points[keep]
        if extra:
            mask = np.vstack((mask, false))
        result.append(mask[np.argsort(mask[:, 0], kind="stable")])
    return tuple(result)


def masks_for(ensemble, reading, goal, draws):
    samples = nominal_samples if ensemble.mode == "nominal" else MODELS["masks"]
    split = np.array_split(draws, len(ensemble.filters))
    return tuple(mask for belief, nodes in zip(ensemble.filters, split)
                 for mask in samples(belief, reading, goal, nodes))


class Ensemble:
    def __init__(self, mode, task):
        self.mode = mode
        _, near, target, label = TASKS[task]
        self.label = label
        self.excluded = (tuple(WORLD.branch_sites) if label else (near, target))
        new = MapFilter if mode == "nominal" else MODELS["filter"]
        self.filters = tuple(new() for _ in range(2 if mode == "reference" else 1))

    def assimilate(self, public, event):
        for j, belief in enumerate(self.filters):
            if j and public.goal[0] in self.excluded:
                belief.advance(public.reading, track=False)
            else:
                belief.observe(WORLD, public.reading, *public.goal, public.mask,
                               event=event, retain=False, track=False)

    def claims(self, public):
        if not len(public.mask):
            return None, 0., []
        likelihood, association = self.filters[0].emission(
            WORLD, public.reading, *public.goal, public.mask)
        priors = np.stack([b.prior_at(public.reading)[0].ravel() for b in self.filters])
        denominator = priors @ likelihood.ravel()
        confidence = priors @ association[:, self.label].reshape(len(public.mask), -1).T
        confidence /= denominator[:, None]
        robust = np.min(confidence, axis=0)
        selected = int(np.argmax(robust))
        return (selected if robust[selected] >= CONFIDENCE else None,
                float(robust[selected]), confidence[:, selected].tolist())

    def kernel(self, reading, goal, nodes):
        masks = masks_for(self, reading, goal, nodes)
        shape = self.filters[0].weights.shape
        likelihood = np.zeros((len(masks), *shape))
        association = np.zeros((3, *likelihood.shape))
        for n, mask in enumerate(masks):
            lik, joint = self.filters[0].emission(WORLD, reading, *goal, mask)
            likelihood[n] = lik
            association[:len(mask), n] = joint[:, self.label]
        priors = np.stack([b.prior_at(reading)[0] for b in self.filters])
        proposal = likelihood.reshape(len(masks), -1) @ priors.mean(axis=0).ravel()
        return Kernel(masks, likelihood, association, proposal)

    def outcomes(self, reading, target_reading, first):
        """Observable first-mask samples with conditional predictive priors."""
        masks = masks_for(self, reading, first, OUTER)
        likelihood = np.stack([
            self.filters[0].emission(WORLD, reading, *first, mask)[0] for mask in masks
        ])
        priors = np.stack([b.prior_at(reading)[0] for b in self.filters])
        new = priors[:, None] * likelihood[None]
        evidence = new.sum(axis=(2, 3))
        proposal = evidence.mean(axis=0)
        weights = evidence / proposal
        weights /= weights.sum(axis=1, keepdims=True)
        posterior = new / evidence[..., None, None]
        if len(priors) == 2 and first[0] in self.excluded:
            posterior[1, :] = priors[1]
        transition = self.filters[0].transition(target_reading.t - reading.t,
                                              .5 * (reading.velocity + target_reading.velocity))
        return posterior @ transition, weights


def quadrature(priors, kernel):
    """Per-model probability of a correct or wrong accepted fresh image."""
    n_models, n_branches = priors.shape[:2]
    weights = priors.reshape(n_models * n_branches, -1)
    num_masks = len(kernel.masks)
    density = weights @ kernel.likelihood.reshape(num_masks, -1).T
    joint = weights @ kernel.association.reshape(3 * num_masks, -1).T
    density = density.reshape(n_models, n_branches, num_masks)
    probabilities = joint.reshape(n_models, n_branches, 3, num_masks) / density[:, :, None]
    robust = probabilities.min(axis=0)
    ordinal = robust.argmax(axis=1)
    confidence = np.take_along_axis(robust, ordinal[:, None], axis=1)[:, 0]
    selected = np.take_along_axis(probabilities, ordinal[None, :, None], axis=2)[:, :, 0]
    importance = density / kernel.proposal[None, None]
    importance /= importance.sum(axis=2, keepdims=True)
    mass = importance * (confidence[None] >= CONFIDENCE)
    return np.stack(((mass * selected).sum(axis=2), (mass * (1. - selected)).sum(axis=2)), axis=-1)


def nominal_turn(path):
    endpoints = MODEL.chart_reference(path.start, path.goal, np.array([0., 1.]), 15.)[:, :3, :3]
    a, b = Rotation.from_matrix(endpoints)
    return float(np.rad2deg((a.inv() * b).magnitude()))


def legal(first, tail):
    return (first.cost_mm + tail.cost_mm <= MAX_TRAVEL_MM
            and TURNS[first.id] + TURNS[tail.id] <= MAX_TURN_DEG
            and first.stroke_mm + tail.stroke_mm <= MAX_STROKE_MM)


def price(path):
    return FIRST_MOVE_PRICE * path.cost_mm + TIME_PRICE * path.duration


def plan(ensemble, graph):
    belief = ensemble.filters[0]
    reading = belief.predicted_reading(belief.reading.t + SLOT)
    terminal = belief.predicted_reading(reading.t + SLOT)
    branches, first_weights = zip(*(ensemble.outcomes(reading, terminal, path.goal)
                                    for path in graph.first))
    priors = np.concatenate(branches, axis=1)
    by_goal = {}
    for goal in sorted({p.goal for tails in graph.tails.values() for p in tails}):
        by_goal[goal] = quadrature(priors, ensemble.kernel(terminal, goal, INNER))
    rows = []
    for i, first in enumerate(graph.first):
        tails = graph.tails[first.id]
        a, b = i * len(OUTER), (i + 1) * len(OUTER)
        probs = np.stack([by_goal[tail.goal][:, a:b] for tail in tails], axis=2)
        motion = np.array([price(tail) + price(first) for tail in tails])
        utility = probs[..., 0] - probs[..., 1] - motion[None, None, :]
        joint = np.einsum("mb,mbt->mt", first_weights[i], utility)
        open_score = float(joint.min(axis=0).max())
        tail_choice = np.argmax(utility.min(axis=0), axis=1)
        idx = np.arange(len(OUTER))
        adaptive = np.einsum("mb,mb->m", first_weights[i], utility[:, idx, tail_choice])
        risk = np.einsum("mb,mb->m", first_weights[i], probs[:, idx, tail_choice, 1])
        correct = np.einsum("mb,mb->m", first_weights[i], probs[:, idx, tail_choice, 0])
        cost = first_weights[i] @ motion[tail_choice]
        rows.append({"goal": list(first.goal), "fixed": open_score, "feedback": adaptive.tolist(),
                     "pred_correct": correct.tolist(), "pred_wrong": risk.tolist(),
                     "pred_motion_cost": cost.tolist(),
                     "tail_views": [list(tail.goal) for tail in tails],
                     "forecast_tail_switches": int(len(set(tail_choice)))})
    baseline = max(range(len(rows)), key=lambda j: rows[j]["fixed"])
    feedback = max(range(len(rows)), key=lambda j: min(rows[j]["feedback"]))
    base = rows[baseline]
    for row in rows:
        row["paired_lower_gain"] = float(np.min(
            np.asarray(row["pred_correct"]) - base["pred_correct"]
            - (np.asarray(row["pred_motion_cost"]) - base["pred_motion_cost"])))
        row["paired_upper_wrong"] = float(np.max(np.asarray(row["pred_wrong"]) - base["pred_wrong"]))
    eligible = [j for j, row in enumerate(rows) if row["paired_upper_wrong"] <= .02
                and row["paired_lower_gain"] > 0.]
    challenge = max(eligible, key=lambda j: rows[j]["paired_lower_gain"], default=baseline)
    return rows, {"MPC": baseline, "B_feedback": feedback, "B_challenge": challenge}


def tail(ensemble, paths):
    reading = ensemble.filters[0].predicted_reading(ensemble.filters[0].reading.t + SLOT)
    priors = np.stack([b.prior_at(reading)[0] for b in ensemble.filters])[:, None]
    predicted = [quadrature(priors, ensemble.kernel(reading, path.goal, INNER))[:, 0] for path in paths]
    utilities = [float((value[:, 0] - value[:, 1]).min()) - price(path)
                 for value, path in zip(predicted, paths)]
    best = int(np.argmax(utilities))
    return paths[best], utilities


def move(case, path, clock, state, cache):
    key = path.id, clock, np.asarray(state).tobytes()
    if key not in cache:
        trace, q = MODEL.execute(path, clock, lambda t: case.true_angle_rate(t)[0], state)
        frame = None
        if trace["legal_completed"]:
            frame = MODELS["render"](WORLD, case, trace["t_end"], *path.goal)
        cost = np.array([trace["travel_mm"], trace["turn_deg"], trace["duration_s"], trace["stroke_mm"]])
        cache[key] = frame, cost, q, trace["t_end"]
    return cache[key]


def start_task(case, task, mode, cache):
    graph = GRAPHS[task]
    frames = WORLD.initial(case)
    ensemble = Ensemble(mode, task)
    for i, frame in enumerate(frames):
        jittered = MODELS["render"](WORLD, case, frame.t, frame.site, frame.view)
        ensemble.assimilate(packet(jittered), i)
    clock = frames[-1].t
    state, _ = MODEL.command(graph.prefix[0], 0., case.true_angle_rate(clock)[0])
    for i, path in enumerate(graph.prefix, start=len(frames)):
        frame, _, state, clock = move(case, path, clock, state, cache)
        if frame is None:
            return None
        ensemble.assimilate(packet(frame), i)
    return ensemble, clock, state


def execute(case, task, initial, first, cache):
    ensemble, clock, state = initial
    start = ensemble.filters
    ensemble.filters = tuple(b.copy() for b in start)
    travel = np.zeros(4)
    records = []
    selected = None
    reason = "motion_failure"
    final = None
    confidence, probabilities = 0., []
    try:
        path = first
        for i in range(2):
            frame, used, state, clock = move(case, path, clock, state, cache)
            travel += used
            if frame is None:
                break
            public = packet(frame)
            records.append({"goal": list(public.goal), "t": public.reading.t,
                            "q": public.reading.q, "velocity": public.reading.velocity,
                            "centroids_mm": public.mask.tolist(), "cost": used.tolist()})
            if i == 0:
                ensemble.assimilate(public, 64)
                paths = GRAPHS[task].tails[first.id]
                path, scores = tail(ensemble, paths)
                records[-1]["tail_scores"] = scores
            else:
                if np.any(travel[[0, 1, 3]] > [MAX_TRAVEL_MM, MAX_TURN_DEG, MAX_STROKE_MM]):
                    reason = "resource_failure"
                    break
                final = frame
                selected, confidence, probabilities = ensemble.claims(public)
                reason = "accepted" if selected is not None else "unconfirmed"
    finally:
        ensemble.filters = start
    correct = selected is not None and final.labels[selected] == ensemble.label
    wrong = selected is not None and not correct
    return {"success": bool(correct), "wrong": bool(wrong), "reason": reason,
            "confidence": confidence, "claim": selected, "model_confidence": probabilities,
            "final_labels_eval_only": None if final is None else list(final.labels),
            "costs": travel.tolist(), "steps": records}


def register_graph(task):
    reference, near, target, _ = TASKS[task]
    prefix = (MODEL.register(WORLD.initial_route[-1], (3, 3), 2.50),
              MODEL.register((3, 3), (reference, 3), SLOT))
    if not all(p.legal for p in prefix):
        raise RuntimeError(f"shared prefix failed registration: {task}")
    candidates = []
    by_first = {}
    for site in (reference, near):
        for view in FIRST_VIEWS:
            first = MODEL.register(prefix[-1].goal, (site, view), SLOT)
            if not first.legal:
                continue
            TURNS[first.id] = nominal_turn(first)
            tails = []
            for v in TAIL_VIEWS:
                end = MODEL.register(first.goal, (target, v), SLOT)
                if not end.legal:
                    continue
                TURNS[end.id] = nominal_turn(end)
                if legal(first, end):
                    tails.append(end)
            if tails:
                by_first[first.id] = tuple(tails)
                candidates.append(first)
    if not all(any(p.goal == (site, 3) for p in candidates) for site in (reference, near)):
        raise RuntimeError(f"reference and near center controls need legal tails: {task}")
    return task, Graph(prefix, tuple(candidates), by_first), MODEL.q_reference


def initialize(repo, samples, fit):
    global WORLD, MODEL, MapFilter, Reading, CONFIDENCE, LOWER, UPPER, IMAGE_SD
    global OUTLIER_SD, OUTLIER_RATE, NEAR_CLUTTER_SD, NEAR_CLUTTER_RATE, OUTER, INNER, TURNS, GRAPHS
    exp = Path(repo) / "NMI_idea" / "experiments"
    y = exp / "ab_tro_y_continuation_20261112"
    sys.path[:0] = [str(y), str(exp / "ab_map_recheck_20261112")]
    from map_model import MapFilter, MapWorld
    import dynamic_scan
    from belief import (Reading, CONFIDENCE, HALF_WIDTH, DEPTH, IMAGE_SD, OUTLIER_SD,
                        OUTLIER_RATE, NEAR_CLUTTER_SD, NEAR_CLUTTER_RATE)
    from mechanics import MotionModel
    from clutter_model import render_observation, FittedMapFilter, simulated_masks
    with np.load(y / "review_inputs" / "POSED_BRANCH.npz", allow_pickle=False) as asset:
        posed = (asset["dense"], asset["s"], asset["daughter"],
                 tuple(map(int, asset["sites"])), int(asset["fork"]))
    with patch.object(dynamic_scan, "posed_branch", return_value=posed):
        WORLD = MapWorld(mode="stress")
    MODEL = MotionModel(WORLD)
    MODELS["render"] = render_observation
    calibration = json.loads(Path(fit).read_text())
    MODELS["filter"] = lambda: FittedMapFilter(calibration=calibration)
    MODELS["masks"] = lambda b, r, g, d: simulated_masks(WORLD, b, r, *g, d)
    LOWER, UPPER = np.array([-HALF_WIDTH, DEPTH[0]]), np.array([HALF_WIDTH, DEPTH[1]])
    OUTER = qmc.Sobol(16, scramble=True, seed=38471).random_base2(samples.bit_length() - 1)
    INNER = qmc.Sobol(16, scramble=True, seed=38479).random_base2(samples.bit_length() + 1)
    TURNS, GRAPHS = {}, {}
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("fork")) as pool:
        for task, graph, references in pool.map(register_graph, TASKS):
            GRAPHS[task] = graph
            MODEL.q_reference.update(references)


def paired(job):
    idx, split, tasks, modes, arms, audit_roots = job
    seed = int(np.random.SeedSequence([SEED_BASES[split], idx]).generate_state(1)[0])
    case = WORLD.case(seed)
    cache = {}
    output = []
    for task in tasks:
        for mode in modes:
            start = start_task(case, task, mode, cache)
            scores, actions = [], {}
            elapsed = 0.
            if start is not None:
                before = time.perf_counter()
                graph = GRAPHS[task]
                scores, actions = plan(start[0], graph)
                elapsed = time.perf_counter() - before
                for arm, site in zip(("Ref_fixed", "Near_fixed"), TASKS[task][:2]):
                    actions[arm] = next(i for i, p in enumerate(graph.first) if p.goal == (site, 3))
            results = {}
            audit = None
            if audit_roots and start is not None:
                for j, path in enumerate(GRAPHS[task].first):
                    results[j] = execute(case, task, start, path, cache)
                audit = [{"first": list(GRAPHS[task].first[j].goal),
                          "success": value["success"], "wrong": value["wrong"],
                          "costs": value["costs"], "reason": value["reason"]}
                         for j, value in results.items()]
            for arm in arms:
                if start is None:
                    result = {"success": False, "wrong": False, "reason": "prefix_failure",
                              "costs": [0., 0., 0., 0.], "steps": []}
                else:
                    j = actions[arm]
                    if j not in results:
                        results[j] = execute(case, task, start, GRAPHS[task].first[j], cache)
                    result = results[j]
                output.append(dict(result, index=idx, seed=seed, task=task,
                                   arm=arm, estimator=mode, forecast=scores,
                                   first_index=actions.get(arm), root_planning_s=elapsed,
                                   evaluator_root_audit=audit if arm == "MPC" else None))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--fit", type=Path, default=Path(__file__).with_name("clutter_fit.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--audit-roots", action="store_true")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--split", choices=SEED_BASES, default="smoke")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    args = parser.parse_args()
    if args.samples < 4 or args.samples & (args.samples - 1):
        raise ValueError("samples must be a power of two")
    args.out.mkdir(parents=True, exist_ok=False)
    config = vars(args) | {
        "seed_base": SEED_BASES[args.split],
        "source_sha256": {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), args.fit, Path(__file__).with_name("clutter_model.py"))},
        "tasks": {name: TASKS[name] for name in args.tasks},
        "cost_caps": [MAX_TRAVEL_MM, MAX_TURN_DEG, 2 * SLOT, MAX_STROKE_MM],
    }
    (args.out / "settings.json").write_text(json.dumps(config, default=str, indent=2) + "\n")
    initialize(args.repo, args.samples, args.fit)
    print("Checked graph", {k: (len(v.first), sum(map(len, v.tails.values())))
                            for k, v in GRAPHS.items()}, flush=True)
    jobs = [(i, args.split, args.tasks, args.modes, args.arms, args.audit_roots) for i in range(args.n)]
    counts = Counter()
    with (args.out / "cases.jsonl").open("w") as out:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("fork")) as pool:
            for i, rows in enumerate(pool.map(paired, jobs), 1):
                for row in rows:
                    out.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                    counts["/".join([row["task"], row["estimator"], row["arm"]])] += row["success"]
                out.flush()
                if i == 1 or i % 8 == 0 or i == len(jobs):
                    print(f"{i}/{len(jobs)} paired cases", flush=True)
    (args.out / "counts.json").write_text(json.dumps(dict(counts), indent=2) + "\n")


if __name__ == "__main__":
    main()
