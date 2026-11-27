"""Post-stop diagnostic for choosing independent corroboration from public evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from refine_rechecks import (
    ARMS, BAND, DYNAMIC, SD, STRESS, VERIFY, DynamicScan, Edges, EvidenceBook,
    Planner, Recheck, build_support, draw_truth, observe,
)


class Corroboration(EvidenceBook):
    def __init__(self, model, z):
        super().__init__(model, gate="predictive", predictive_z=z)
        self.quality = {}
        self.fresh = False

    def _record_evidence(self, site, values, assoc, joint, joint_probability,
                         labels, model_terms, pre_mean, pre_covariance):
        before = {i: len(self.history[i]) for i, _ in self.model.loc[site]}
        super()._record_evidence(site, values, assoc, joint, joint_probability,
                                 labels, model_terms, pre_mean, pre_covariance)
        for i, label in self.model.loc[site]:
            if len(self.history[i]) == before[i]:
                continue
            terms, _ = self.history[i][-1]
            assignment = joint_probability if label == 2 else float(np.max(assoc[label]))
            inlier = 1.
            for h, g, value in terms:
                error = value - g - h @ pre_mean
                latent_variance = h @ pre_covariance @ h
                variances = latent_variance + np.array([SD**2, 16.])
                weight = np.array([1. - STRESS.jump_rate, STRESS.jump_rate])
                density = weight * np.exp(-error**2 / (2. * variances)) / np.sqrt(variances)
                inlier *= density[0] / density.sum()
            self.quality[id(terms)] = (float(assignment * inlier), self.fresh)

    def copy(self):
        result = type(self)(self.model, self.predictive_z)
        # Terms are shared read-only so their pre-update scores retain identity.
        result.history = [events.copy() for events in self.history]
        result.events = result.history
        result.quality = self.quality.copy()
        result.fresh = self.fresh
        return result

    def rank(self, events):
        def priority(event):
            probability, fresh = self.quality[id(event[0])]
            return fresh, probability

        return sorted(events, key=priority, reverse=True)


def replay(model, prefix, row, z):
    book = Corroboration(model, z)
    belief = np.zeros(model.scan.dim), model.scan.prior.copy()
    belief[1][-1, -1] = STRESS.slip_sd_mm**2
    previous = None
    for steps, key in ((prefix, "observation_mm"), (row["steps"], "centroids_mm")):
        for step in steps:
            pose, action = step["pose"], tuple(step["action"])
            belief = book.assimilate(belief, pose, action, step[key], step["eval_only_labels"])
            previous = (pose, *action)
        book.fresh = True
    return book, belief, previous


def evaluate(book, belief, truth, rank):
    _, variance, ready, inliers = book.status(belief)
    best = [book.rank(events)[:VERIFY] if rank else events[-VERIFY:] for events in inliers]
    wrong = [i for i, events in enumerate(best) if len(events) >= VERIFY
             and not all(correct for _, correct in events)]
    maximum_error = float(np.max(np.abs(book.model.query @ (belief[0] - truth))))
    return {
        "online_stop": bool(np.all(ready)), "wrong": wrong, "map_max_error_mm": maximum_error,
        "max_band_mm": float(book.model.gamma * np.sqrt(variance.max())),
        "complete": bool(np.all(ready) and not wrong and maximum_error <= BAND),
    }


def verify(planner, base, prefix, truth, book, belief, previous, mode, count):
    model = planner.model
    start = int(np.random.default_rng(np.random.SeedSequence([base["seed"], 81])).integers(len(model.scan.moving)))
    visited, records = set(), []
    for k in range(min(count, 80 - len(base["steps"]))):
        counts, variance, ready, inliers = book.status(belief)
        pose = model.scan.pose(STRESS, len(prefix) + len(base["steps"]) + k, start)
        risk = np.zeros(model.n)
        for i, events in enumerate(inliers):
            if i in visited:
                continue
            pair = [book.quality[id(terms)] for terms, _ in book.rank(events)[:VERIFY]]
            weakest = min((score for score, _ in pair), default=0.)
            prefix_dependence = 1. - sum(is_fresh for _, is_fresh in pair) / VERIFY
            risk[i] = (1. - weakest) * (1. + prefix_dependence)
            risk[i] += .04 * np.sqrt(variance[i] / model.var_limit)
            risk[i] += float(not ready[i])
        # Choose a contact that would add one corroborating witness at a
        # publicly risky target. All routes start at the actual last contact.
        bank = planner.actions[pose]
        sites = np.asarray([i for i, _ in model.scan.targets])
        distances = np.linalg.norm(planner.positions[pose] - model.scan.views[previous], axis=1)
        proposals = []
        for target in np.flatnonzero(risk):
            subset = np.flatnonzero(bank[:, 0] == sites[target])
            if not len(subset):
                continue
            needs = counts.copy()
            needs[target] = min(needs[target], VERIFY - 1)
            useful = planner.score(pose, previous, belief, needs, ARMS["B"], subset)
            for j, value in zip(subset, useful):
                score = risk[target] * value if mode == "targeted" else -distances[j]
                proposals.append((float(score), int(target), int(j)))
        chosen = next((
            (t, j) for _, t, j in sorted(proposals, reverse=True)
            if model.edge.allow(previous, (pose, *bank[j]))
        ), None)
        if chosen is None:
            break
        target, j = chosen
        visited.add(target)
        action = tuple(bank[j])
        motion = planner.motion(previous, pose, action)
        step = len(base["steps"]) + k
        values, labels = observe(model, pose, action, truth, base["seed"], step)
        belief = book.assimilate(belief, pose, action, values, labels)
        previous = (pose, *action)
        records.append({
            "pose": int(pose), "action": list(map(int, action)), "target": target,
            "risk": float(risk[target]), "length_m": motion.length_m, "path_m": motion.points.tolist(),
            "centroids_mm": values, "eval_only_labels": labels,
        })
    return evaluate(book, belief, truth, rank=True) | {"extra": records}


def run(base, planner, truth, prefix, z, count):
    book, belief, previous = replay(planner.model, prefix, base, z)
    old = evaluate(book, belief, truth, rank=False)
    if not (old["complete"] == base["complete"]
            and old["wrong"] == base["incorrect_witnesses"]
            and np.isclose(old["map_max_error_mm"], base["map_max_error_mm"], rtol=0., atol=1.e-10)):
        raise ValueError(f"mismatched evidence replay for {base['seed']}")
    ranked = evaluate(book, belief, truth, rank=True)
    outcomes = {"original": old | {"extra": []}, "ranked": ranked | {"extra": []}}
    for mode in ("targeted", "nearest"):
        outcomes[mode] = (verify(planner, base, prefix, truth, book.copy(), belief, previous, mode, count)
                          if old["online_stop"] else ranked | {"extra": []})
    return {"seed": base["seed"], "base_arm": base["arm"], "base_frames": len(base["steps"]),
            "base_m": base["recheck_path_mm"] / 1000., "outcomes": outcomes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--extra", type=int, default=3)
    parser.add_argument("--arm", default="B", choices=["locked", "A", "B", "B_no_fisher", "B_endpoint"])
    args = parser.parse_args()
    with np.load(DYNAMIC / "DYNAMIC_GEOMETRY_FROZEN.npz") as geometry:
        scan = DynamicScan(geometry)
        support = build_support(scan, geometry)
    model = Recheck(scan, Edges(scan))
    planner = Planner(model, travel_weight=1.4, n_predictive=8, n_first=3,
                      switch_margin=.006, predictive_z=2.)
    result = []
    for shard in args.runs:
        config = json.loads((shard / "CONFIG.json").read_text())
        for line in (shard / "RUNS.jsonl").read_text().splitlines():
            base = json.loads(line)
            if base["arm"] != args.arm:
                continue
            truth, _ = draw_truth(scan, base["seed"], support)
            prefix = scan.run(STRESS, base["seed"], "A", truth)["steps"]
            row = run(base, planner, truth, prefix, config["predictive_z"], args.extra)
            result.append(row)
            print(row["seed"], {key: val["complete"] for key, val in row["outcomes"].items()}, flush=True)
    args.out.write_text(json.dumps({"extra_cap": args.extra, "rows": result}, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
