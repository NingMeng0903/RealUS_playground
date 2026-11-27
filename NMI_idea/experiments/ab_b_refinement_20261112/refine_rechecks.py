"""Paired closed-loop rechecks using public evidence and sampled checked exits."""
from __future__ import annotations

import argparse
from copy import copy
from dataclasses import dataclass
from functools import lru_cache, partial
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "ab_fisher_continuation_20261112"
sys.path.insert(0, str(BASE))
from certification import (  # noqa: E402
    BAND, PD, SD, STRESS, VERIFY, WIDTH, DYNAMIC, DynamicScan, Edges, Recheck,
    build_support, draw_truth,
)
from evidence import EvidenceBook  # noqa: E402


@dataclass(frozen=True)
class Arm:
    gate: str
    mode: str
    fisher: float = .42
    travel_scale: float = 1.
    learned_primitive: bool = True
    keep_prefix_witnesses: bool = True


ARMS = {
    "old_A_fisher": Arm("legacy", "legacy_greedy"),
    "old_fixed": Arm("legacy", "legacy"),
    "gate_only": Arm("predictive", "legacy"),
    "A": Arm("predictive", "greedy"),
    "locked": Arm("predictive", "locked"),
    "B": Arm("predictive", "live"),
    "B_once": Arm("predictive", "once"),
    "B_no_fisher": Arm("predictive", "live", fisher=0.),
    "B_low_travel_cost": Arm("predictive", "live", travel_scale=.4),
    "B_endpoint": Arm("predictive", "live", learned_primitive=False),
    "fresh_locked": Arm("predictive", "locked", keep_prefix_witnesses=False),
    "fresh_B": Arm("predictive", "live", keep_prefix_witnesses=False),
}


@dataclass
class Motion:
    points: np.ndarray
    skill: str
    length_m: float


@dataclass
class Program:
    options: np.ndarray
    locked: dict[bool, int]
    predicted_gain: float
    first_progress_probability: float


@dataclass
class Choice:
    index: int
    locked_index: int
    usable: bool
    gain: float
    predicted_gain: float


class Planner:
    def __init__(self, model, *, travel_weight, n_predictive, n_first, switch_margin, predictive_z):
        self.model = model
        self.scan = model.scan
        self.endpoint_scan = copy(self.scan)
        self.endpoint_scan.primitive = copy(self.scan.primitive)
        self.endpoint_scan.primitive.path = partial(self.scan.primitive.path, learned=False)
        self.travel_weight = travel_weight
        self.n_predictive = n_predictive
        self.n_first = n_first
        self.switch_margin = switch_margin
        self.predictive_z = predictive_z
        self.actions = {p: np.asarray(actions, dtype=int) for p, actions in model.action_bank.items()}
        self.index = {p: {tuple(a): i for i, a in enumerate(actions)} for p, actions in self.actions.items()}
        self.models = {}
        self.targets = {}
        self.positions = {}
        self.q = {}
        self.margins = {}
        for pose, actions in self.actions.items():
            n, dim = len(actions), self.scan.dim
            rows, bases = np.zeros((n, 2, dim)), np.zeros((n, 2))
            single = np.full((n, 2), -1, dtype=int)
            both = np.full(n, -1, dtype=int)
            for index, (site, view) in enumerate(actions):
                for j, (row, base) in enumerate(model.model_cache[pose, site, view]):
                    rows[index, j], bases[index, j] = row, base
                for target, label in model.loc[site]:
                    if label == 2:
                        both[index] = target
                    else:
                        single[index, label] = target
            i, j = actions.T
            self.models[pose] = rows, bases
            self.targets[pose] = single, both
            self.positions[pose] = self.scan.views[pose, i, j]
            self.q[pose] = self.scan.q[pose, i, j]
            self.margins[pose] = self.scan.dird[pose, i, j]

    @lru_cache(maxsize=46000)
    def motion(self, previous, pose, action, learned=True):
        scan = self.scan if learned else self.endpoint_scan
        points, skill = scan.curve(previous, pose, action)
        length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
        return Motion(points, skill, float(length))

    def _costs(self, previous, pose, indices, arm, exact):
        if previous is None:
            return np.zeros(len(indices))
        position = self.scan.views[previous]
        q = self.scan.q[previous]
        if exact:
            distances = np.array([
                self.motion(previous, pose, tuple(self.actions[pose][i]), arm.learned_primitive).length_m
                for i in indices
            ])
        else:
            distances = np.linalg.norm(self.positions[pose][indices] - position, axis=1)
        dq = self.q[pose][indices] - q
        return (self.travel_weight * arm.travel_scale * distances
                + .00294 * np.linalg.norm(dq[:, 1:], axis=1) + .0105 * np.abs(dq[:, 0]))

    def score(self, pose, previous, belief, counts, arm, indices=None, *, exact=False):
        """Witness deficit, precision deficit, optional task Fisher, and exit cost."""
        if indices is None:
            indices = np.arange(len(self.actions[pose]))
        mean, cov = belief
        rows, bases = (v[indices] for v in self.models[pose])
        single, both = (v[indices] for v in self.targets[pose])
        prior = rows @ cov
        sigma2 = np.einsum("nld,nld->nl", rows, prior) + SD**2
        sigma = np.sqrt(sigma2)
        expected = bases + rows @ mean
        # Nominal component capture within the pre-update witness gate.
        a = np.maximum((-WIDTH - expected) / sigma, -self.predictive_z)
        b = np.minimum((WIDTH - expected) / sigma, self.predictive_z)
        hit = PD * (1. - STRESS.jump_rate) * np.maximum(ndtr(b) - ndtr(a), 0.)
        paired = both >= 0
        two_lumens = np.any(rows[:, 1] != 0., axis=1)
        diff = rows[:, 1] - rows[:, 0]
        separation_sd = np.sqrt(np.einsum("nd,nd->n", diff @ cov, diff) + 2. * SD**2)
        # A cheap association proxy only for scoring the next view. Predicted
        # and executed observations still use the full public mixture update.
        separation = np.abs(expected[:, 1] - expected[:, 0]) / separation_sd
        pair_confidence = 2. * ndtr(separation / 2.) - 1.
        hit *= np.where(two_lumens, .5 + .5 * pair_confidence, 1.)[:, None]
        joint_hit = hit[:, 0] * hit[:, 1] * pair_confidence
        H, V = self.model.worst_queries(cov)
        need = (VERIFY - counts) / VERIFY
        gap = np.clip(np.log(np.maximum(V, 1.e-8) / self.model.var_limit), 0., 3.)
        imprecise = V > self.model.var_limit
        local = need + .16 * (gap + imprecise)
        witness = np.sum(hit * np.where(single >= 0, local[single], 0.), axis=1)
        witness += np.where(paired, joint_hit * local[both], 0.)

        if arm.fisher:
            # Expected rank-one precision benefit for each useful Gaussian
            # hit. Each conditional gain is evaluated before weighting by the
            # hit probability; this is a task-direction proxy, not exact Fisher
            # for an unordered two-lumen image.
            target_weights = (.08 * need + gap + .16 * imprecise) / np.maximum(V, 1.e-8)
            contraction = (prior @ H.T)**2 / sigma2[:, :, None]
            information = np.sum(hit * (contraction @ target_weights)
                                 * np.any(rows != 0., axis=2), axis=1)
            witness += arm.fisher * information
        # A moving acquisition trades against another acquisition, not
        # against collecting nothing. A fixed additive travel penalty made
        # near-threshold progress unattractive even when the task was open.
        return witness / (1. + self._costs(previous, pose, indices, arm, exact))

    def _within_budget(self, previous, pose, index, remaining, arm):
        action = tuple(self.actions[pose][index])
        if not self.model.edge.allow(previous, (pose, *action)):
            return False
        return (not np.isfinite(remaining)
                or self.motion(previous, pose, action, arm.learned_primitive).length_m <= remaining + 1.e-10)

    def _top(self, score, pose, previous, limit, remaining, arm, indices=None):
        if indices is None:
            indices = np.arange(len(score))
        feasible = []
        for i in indices[np.argsort(-score[indices], kind="stable")]:
            if self._within_budget(previous, pose, int(i), remaining, arm):
                feasible.append(int(i))
                if len(feasible) >= limit:
                    break
        return feasible

    def _outcomes(self, book, pose, index, belief, counts):
        from predictive import predictive_sets

        action = tuple(self.actions[pose][index])
        model = self.model.model_cache[(pose, *action)]
        samples = predictive_sets(
            belief, model, half_width=WIDTH, p_detect=PD,
            noise_mixture=STRESS.noise_mixture, clutter_rate=STRESS.clutter_rate,
            count=self.n_predictive,
        )
        for values, weight in samples:
            predicted = book.copy()
            updated = predicted.assimilate(belief, pose, action, values)
            after, _, ready, _ = predicted.status(updated)
            yield updated, after, bool(np.any(after > counts)), bool(np.all(ready)), weight

    def _plan_suffix(self, book, belief, counts, pose, index, next_pose, arm, remaining):
        action = tuple(self.actions[pose][index])
        exit_state = (pose, *action)
        outcomes = list(self._outcomes(book, pose, index, belief, counts))
        base = self.score(next_pose, exit_state, belief, counts, arm)
        pool = set(self._top(base, next_pose, exit_state, 8, remaining, arm))
        nearby = np.flatnonzero(np.abs(self.actions[next_pose][:, 0] - action[0]) <= 1)
        pool.update(self._top(base, next_pose, exit_state, 4, remaining, arm, nearby))
        for posterior, updated_counts, _, ready, _ in outcomes:
            if ready:
                continue
            forecast = self.score(next_pose, exit_state, posterior, updated_counts, arm)
            pool.update(self._top(forecast, next_pose, exit_state, 2, remaining, arm))
        if not pool:
            return None, 0.
        candidates = np.array(sorted(pool))
        values = np.stack([
            np.zeros(len(candidates)) if ready else self.score(
                next_pose, exit_state, posterior, updated_counts, arm, candidates, exact=True,
            )
            for posterior, updated_counts, _, ready, _ in outcomes
        ])
        weights = np.array([o[4] for o in outcomes])
        groups = np.array([o[2] for o in outcomes], dtype=bool)
        default = int(np.argmax(weights @ values))
        locked, expectation, adapted = {}, 0., 0.
        for category in (False, True):
            select = groups == category
            if np.any(select):
                expected = weights[select] @ values[select]
                choice = int(np.argmax(expected))
                expectation += float(expected[choice])
                adapted += float(weights[select] @ np.max(values[select], axis=1))
            else:
                choice = default
            locked[category] = int(candidates[choice])
        return Program(candidates, locked, max(adapted - expectation, 0.),
                       float(weights @ groups)), expectation

    def plan(self, book, pose, next_pose, previous, belief, counts, arm, remaining, last):
        scores = self.score(pose, previous, belief, counts, arm)
        candidates = np.asarray(self._top(scores, pose, previous, 7, remaining, arm))
        if len(candidates) == 0:
            return None, None
        exact = self.score(pose, previous, belief, counts, arm, candidates, exact=True)
        ranked = np.argsort(-exact, kind="stable")
        first = int(candidates[ranked[0]])
        if last or arm.mode == "greedy":
            return first, None
        best = -np.inf
        selected = None
        for k in ranked[:self.n_first]:
            index = int(candidates[k])
            action = tuple(self.actions[pose][index])
            distance = self.motion(previous, pose, action, arm.learned_primitive).length_m
            program, value = self._plan_suffix(
                book, belief, counts, pose, index, next_pose, arm, remaining - distance,
            )
            # Same first decision for locked and live at the same history.
            objective = float(exact[k] + .8 * value)
            if objective > best:
                best, first, selected = objective, index, program
        return first, selected

    def choose(self, program, pose, previous, belief, before, after, arm, switched):
        usable = bool(np.any(after > before))
        locked = program.locked[usable]
        index, gain = locked, 0.
        if arm.mode == "live" or (arm.mode == "once" and not switched):
            scores = self.score(pose, previous, belief, after, arm, program.options, exact=True)
            best = int(np.argmax(scores))
            anchor = int(np.flatnonzero(program.options == locked)[0])
            gain = float(scores[best] - scores[anchor])
            if gain > self.switch_margin:
                index = int(program.options[best])
        return Choice(index, locked, usable, gain, program.predicted_gain)


def observe(model, pose, action, evaluator_truth, seed, step):
    """Simulator only. IDs are returned for the offline outcome evaluator."""
    rows = model.model_cache[(pose, *action)]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 741, step]))
    sigma = np.where(rng.random(len(rows)) < STRESS.jump_rate, 4., SD)
    values = np.array([base + row @ evaluator_truth for row, base in rows])
    values += rng.normal(0., sigma)
    detected = rng.random(len(rows)) < PD
    pairs = [(float(v), i) for i, v in enumerate(values) if abs(v) <= WIDTH and detected[i]]
    if rng.random() < STRESS.clutter_rate:
        pairs.append((float(rng.uniform(-WIDTH, WIDTH)), -1))
    pairs.sort()
    return [v for v, _ in pairs], [i for _, i in pairs]


def run_rechecks(planner, prefix, seed, truth, name, cap, path_cap, predictive_z):
    model, scan, arm = planner.model, planner.scan, ARMS[name]
    book = EvidenceBook(model, gate=arm.gate, predictive_z=predictive_z)
    belief = np.zeros(scan.dim), scan.prior.copy()
    belief[1][-1, -1] = STRESS.slip_sd_mm**2
    start = int(np.random.default_rng(np.random.SeedSequence([seed, 81])).integers(len(scan.moving)))
    previous = None
    for s in prefix:
        pose, action = int(s["pose"]), tuple(s["action"])
        belief = book.assimilate(belief, pose, action, s["observation_mm"], s["eval_only_labels"])
        previous = (pose, *action)
    if not arm.keep_prefix_witnesses:
        book = EvidenceBook(model, gate=arm.gate, predictive_z=predictive_z)
    counts, _, ready, _ = book.status(belief)
    initial = int(ready.sum())
    scheduled = None
    old_scheduled = None
    switched = False
    steps = []
    distance = 0.

    for k in range(cap):
        if np.all(ready):
            break
        pose = scan.pose(STRESS, len(prefix) + k, start)
        next_pose = scan.pose(STRESS, len(prefix) + k + 1, start)
        remaining = path_cap - distance
        plan, old_plan, tail = None, None, None
        is_tail = scheduled is not None or old_scheduled is not None
        if scheduled is not None:
            tail, scheduled = scheduled, None
            action = tuple(planner.actions[pose][tail.index])
        elif old_scheduled is not None:
            action, old_scheduled = old_scheduled, None
        elif arm.mode.startswith("legacy"):
            strategy = "A_fisher" if arm.mode == "legacy_greedy" else "fixed"
            action, old_plan = model.plan(pose, next_pose, previous, *belief, counts, strategy)
        else:
            index, plan = planner.plan(
                book, pose, next_pose, previous, belief, counts, arm, remaining, k + 1 == cap,
            )
            action = tuple(planner.actions[pose][index]) if index is not None else None
        if action is None:
            break
        if not model.edge.allow(previous, (pose, *action)):
            raise RuntimeError("unchecked exit-to-contact edge")
        motion = planner.motion(previous, pose, action, arm.learned_primitive)
        if motion.length_m > remaining + 1.e-10:
            break
        distance += motion.length_m
        values, labels = observe(model, pose, action, truth, seed, k)
        before = counts
        belief = book.assimilate(belief, pose, action, values, labels)
        counts, _, ready, _ = book.status(belief)
        previous = (pose, *action)
        changed = tail is not None and tail.index != tail.locked_index
        switched |= changed
        if plan is not None:
            scheduled = planner.choose(plan, next_pose, previous, belief, before, counts, arm, switched)
        if old_plan is not None:
            old_scheduled = old_plan["tails"][old_plan["fixed"] if values else old_plan["miss"]]
        steps.append({
            "pose": int(pose), "action": list(map(int, action)), "tail": is_tail,
            "centroids_mm": values, "eval_only_labels": labels,
            "path_m": motion.points.tolist(), "path_skill": motion.skill,
            "length_m": motion.length_m, "cumulative_m": distance,
            "changed_suffix": bool(changed), "evidence_gains": np.flatnonzero(counts > before).tolist(),
            "n_ready": int(ready.sum()),
            "tail_choice": None if tail is None else {
                "locked": list(map(int, planner.actions[pose][tail.locked_index])),
                "usable": tail.usable, "gain": tail.gain,
                "predicted_gain": tail.predicted_gain,
            },
        })
    mean, covariance = belief
    _, V, ready, inliers = book.status(belief)
    wrong = [i for i, events in enumerate(inliers) if len(events) >= VERIFY
             and not all(e[1] for e in events[-VERIFY:])]
    error = model.query @ (mean - truth)
    stop = bool(np.all(ready))
    return {
        "seed": seed, "arm": name, "prefix_acquisitions": len(prefix),
        "initial_certified": initial, "certified": int(ready.sum()), "online_stop": stop,
        "complete": bool(stop and not wrong and np.max(np.abs(error)) <= BAND),
        "incorrect_witnesses": wrong, "map_max_error_mm": float(np.max(np.abs(error))),
        "map_rmse_mm": float(np.sqrt(np.mean(error**2))),
        "max_band_mm": float(model.gamma * np.sqrt(np.max(V))),
        "recheck_path_mm": 1000. * distance, "steps": steps,
    }


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2411202619)
    parser.add_argument("--cap", type=int, default=80)
    parser.add_argument("--path-cap", type=float, default=np.inf, help="optional executed meters")
    parser.add_argument("--travel-weight", type=float, default=1.4)
    parser.add_argument("--predictive-z", type=float, default=2.)
    parser.add_argument("--n-predictive", type=int, default=8)
    parser.add_argument("--n-first", type=int, default=3)
    parser.add_argument("--switch-margin", type=float, default=.006)
    parser.add_argument("--arms", nargs="+", choices=list(ARMS),
                        default=["old_A_fisher", "old_fixed", "gate_only", "locked", "B", "B_no_fisher"])
    args = parser.parse_args()

    geometry = DYNAMIC / "DYNAMIC_GEOMETRY_FROZEN.npz"
    with np.load(geometry) as data:
        scan = DynamicScan(data)
        support = build_support(scan, data)
    planner = Planner(Recheck(scan, Edges(scan)), travel_weight=args.travel_weight,
                      n_predictive=args.n_predictive, n_first=args.n_first,
                      switch_margin=args.switch_margin, predictive_z=args.predictive_z)
    args.out.mkdir(parents=True, exist_ok=False)
    configuration = vars(args) | {
        "out": str(args.out), "path_cap": args.path_cap if np.isfinite(args.path_cap) else None,
        "geometry_sha256": sha(geometry),
        "source_sha256": {p.name: sha(p) for p in [
            HERE / "refine_rechecks.py", HERE / "evidence.py",
            BASE / "certification.py", DYNAMIC / "dynamic_scan.py", DYNAMIC / "stress_observation.py",
        ] + ([HERE / "predictive.py"] if any(ARMS[n].mode in {"locked", "live", "once"} for n in args.arms) else [])},
    }
    (args.out / "CONFIG.json").write_text(json.dumps(configuration, indent=2) + "\n")
    with (args.out / "RUNS.jsonl").open("w") as output:
        for k in range(args.start, args.start + args.n):
            seed = int(np.random.SeedSequence([args.seed, k]).generate_state(1)[0])
            truth, rejects = draw_truth(scan, seed, support)
            prefix = scan.run(STRESS, seed, "A", truth)["steps"]
            for name in args.arms:
                row = run_rechecks(planner, prefix, seed, truth, name, args.cap,
                                   args.path_cap, args.predictive_z)
                row["truth_rejections"] = rejects
                output.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                output.flush()
                changes = sum(s["changed_suffix"] for s in row["steps"])
                print(k, seed, name, row["complete"], len(row["steps"]),
                      round(row["recheck_path_mm"]), changes, flush=True)


if __name__ == "__main__":
    main()
