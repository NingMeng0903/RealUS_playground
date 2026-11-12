"""Frozen, paired knee-only association and recheck experiment."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from belief import CONFIDENCE, Filter, Reading
from knee_world import GEOMETRY, World
from policy import fast_behavior_scores, proposal_features, rule_scores, terminal_score, travel_cost


HERE = Path(__file__).resolve().parent
BUDGET = 3
BASES = {"train": 61842721, "dev": 62558311, "test": 91412639}
METHODS = ("fqi", "cql", "sac")
TRAINING_SEEDS = (7, 29, 61)
TRAINING_CASES = (128, 512)
QUERY_DELAY = 5.5
HORIZONS = (.5, 1.)
TRANSITION_KEYS = ("features", "mask", "action", "reward", "next_features", "next_mask", "done")
CORE_FILES = ("belief.py", "knee_world.py", "policy.py", "experiment.py", "learners.py", "PROTOCOL.txt")


@dataclass
class State:
    belief: Filter
    labels: dict
    previous: object
    frames: tuple
    event: int

    def updated(self, world, frame, *, retain):
        belief = self.belief.copy()
        reading = Reading(frame.t, frame.q, frame.velocity)
        belief.observe(world, reading, frame.site, frame.view, frame.centroids,
                       event=self.event, retain=retain)
        labels = self.labels.copy()
        if retain:
            for j, label in enumerate(frame.labels):
                labels[(self.event, j)] = int(label)
        return State(belief, labels, frame, (*self.frames, frame), self.event + 1)


def seed_at(split, index):
    return int(np.random.SeedSequence([BASES[split], index]).generate_state(1)[0])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepared(world, case, *, latent=True):
    state = State(Filter(latent=latent), {}, None, (), 0)
    for frame in world.initial(case):
        state = state.updated(world, frame, retain=world.identity_site(frame.site))
    if not state.frames:
        raise RuntimeError("rule first sweep produced no captures")
    return state


def prefix_hash(state):
    payload = [[frame.t, frame.q, frame.velocity, frame.site, frame.view,
                np.asarray(frame.centroids).tolist()] for frame in state.frames]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def apply(world, case, state, action):
    frame = world.observe(case, action)
    if frame.t <= state.previous.t:
        raise ValueError("non-positive physical action duration")
    return state.updated(world, frame, retain=world.identity_site(action.site))


def current_identity(state):
    keys, confidence = state.belief.decisions()
    confirmed = confidence >= CONFIDENCE
    labels = [state.labels.get(k, -2) for k in keys]
    correct = [bool(confirmed[i] and labels[i] == i) for i in range(2)]
    wrong = [bool(confirmed[i] and labels[i] != i) for i in range(2)]
    return keys, confidence, labels, correct, wrong


def identity_outcome(prefix, state):
    initial_keys, initial_conf = prefix.belief.decisions()
    keys, confidence, labels, correct, wrong = current_identity(state)
    wrong_prefix = [bool(key is not None and prefix.labels[key] != i)
                    for i, key in enumerate(initial_keys)]
    verified_prefix_wrong = [
        bool(wrong_prefix[i] and initial_conf[i] >= CONFIDENCE) for i in range(2)
    ]
    # Revisit the posterior probability of the same historical witness.
    initial_marginal = {w.key: prefix.belief.probability(w) for w in prefix.belief.witnesses}
    revised = {w.key: state.belief.probability(w) for w in state.belief.witnesses}
    old_reassociated = []
    for i, key in enumerate(initial_keys):
        if not wrong_prefix[i]:
            old_reassociated.append(False)
            continue
        truth = prefix.labels[key] if prefix.labels[key] >= 0 else 2
        old_reassociated.append(bool(np.argmax(initial_marginal[key]) != truth
                                     and np.argmax(revised[key]) == truth))
    return {
        "correct": correct, "wrong": wrong, "confidence": confidence.tolist(),
        "witness_keys": keys, "witness_labels": labels,
        "wrong_clutter": any(flag and labels[i] == -1 for i, flag in enumerate(wrong)),
        "wrong_branch": any(flag and labels[i] == 1-i for i, flag in enumerate(wrong)),
        "both_correct": all(correct),
        "wrong_accept": any(wrong),
        "confirmed_labels": int(np.sum(confidence >= CONFIDENCE)),
        "prefix_provisional_wrong": int(sum(wrong_prefix)),
        "prefix_confident_wrong": int(sum(verified_prefix_wrong)),
        "prefix_confident_wrong_case": any(verified_prefix_wrong),
        "prefix_reassociated": int(sum(old_reassociated)),
        "prefix_wrong_replaced": sum(wrong_prefix[i] and correct[i] for i in range(2)),
        "prefix_clutter_replaced": sum(wrong_prefix[i] and correct[i] and prefix.labels[key] == -1
                                       for i, key in enumerate(initial_keys)),
        "prefix_branch_replaced": sum(wrong_prefix[i] and correct[i] and prefix.labels[key] == 1-i
                                      for i, key in enumerate(initial_keys)),
        "prefix_confident_wrong_replaced": sum(verified_prefix_wrong[i] and correct[i] for i in range(2)),
        "new_confident_wrong": sum(wrong[i] and not verified_prefix_wrong[i] for i in range(2)),
    }


def forecast_outcome(world, case, state, time):
    mean, sd = state.belief.forecast(world, time, world.query_sites)
    true = world.truth_world(case, time, world.query_sites)
    error = mean - true
    return {
        "t": float(time), "rmse_mm": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
        "max_mm": float(np.max(np.linalg.norm(error, axis=1))),
        "coordinate_90_coverage": float(np.mean(np.abs(error) <= 1.6448536269514722 * sd)),
        "mean_xyz_mm": mean.tolist(), "sd_xyz_mm": sd.tolist(), "truth_xyz_mm": true.tolist(),
    }


def evaluate_outcome(world, case, prefix, state, actions, *, trace=False):
    identity = identity_outcome(prefix, state)
    t = state.previous.t
    anchor = prefix.previous.t + QUERY_DELAY
    if anchor < t:
        raise ValueError("forecast anchor earlier than last allowed acquisition")
    current = forecast_outcome(world, case, state, t)
    forecasts = [forecast_outcome(world, case, state, t + h) for h in HORIZONS]
    fixed = forecast_outcome(world, case, state, anchor)
    mean, sd = state.belief.stats()
    z_true = world.true_z(case, t)
    row = {
        **identity,
        "strict": bool(identity["both_correct"] and current["max_mm"] <= 3.),
        "current": current, "future": forecasts, "common_clock": fixed,
        "material_within_3mm": current["max_mm"] <= 3.,
        "z_error_mm": float(mean[2] - z_true),
        "z_estimate_mm": float(mean[2]),
        "z_true_mm": float(z_true),
        "z_sd_mm": float(sd[2]),
        "elapsed_s": float(t - prefix.previous.t),
        "travel_mm": float(sum(a.distance_mm for a in actions)),
        "captures": len(actions),
        "actions": [a.kind for a in actions],
        "prefix_sha256": prefix_hash(prefix),
    }
    if trace:
        row["trace"] = [{
            "t": float(frame.t), "q": float(frame.q), "site": int(frame.site), "view": int(frame.view),
            "probe_xyz_mm": world.probe_world(frame.q, frame.site, frame.view).tolist(),
            "centroids_mm": np.asarray(frame.centroids).tolist(),
        } for frame in state.frames]
    return row


def terminal_reward(world, case, state):
    _, _, _, correct, wrong = current_identity(state)
    t = state.previous.t
    forecast = forecast_outcome(world, case, state, t + HORIZONS[0])
    current = forecast_outcome(world, case, state, t)
    strict = all(correct) and current["max_mm"] <= 3.
    return terminal_score(all(correct), strict, sum(correct), sum(wrong), forecast["rmse_mm"])


def proposals(world, state, remaining):
    actions = world.candidates(state.belief.reading, state.previous)
    features, mask = proposal_features(world, state.belief, actions, remaining, BUDGET)
    if not np.any(mask):
        raise RuntimeError("no legal recheck action")
    return actions, features, mask


def generate_one(world, case, prefix, *, rng):
    """Execute every first legal action from one observed prefix."""
    first, _, mask = proposals(world, prefix, BUDGET)
    transitions, branches = [], []
    for selected in np.flatnonzero(mask):
        state = prefix
        executed = []
        for step in range(BUDGET):
            remaining = BUDGET - step
            actions, features, mask = proposals(world, state, remaining)
            if step == 0:
                slot = int(selected)
                propensity = 1. / int(mask.sum())
            else:
                preferred = int(np.argmax(fast_behavior_scores(features, mask)))
                uniform = np.flatnonzero(mask)
                slot = int(rng.choice(uniform)) if rng.random() < .55 else preferred
                propensity = .55 / len(uniform) + .45 * (slot == preferred)
            action = actions[slot]
            state = apply(world, case, state, action)
            executed.append(action)
            done = remaining == 1
            reward = -travel_cost(action)
            if done:
                reward += terminal_reward(world, case, state)
                next_features = np.zeros_like(features)
                next_mask = np.zeros_like(mask)
            else:
                _, next_features, next_mask = proposals(world, state, remaining - 1)
            transitions.append({
                "features": features, "mask": mask, "action": slot, "reward": reward,
                "next_features": next_features, "next_mask": next_mask, "done": done,
                "first_slot": int(selected), "step": step, "propensity": propensity,
            })
        branches.append(evaluate_outcome(world, case, prefix, state, executed))
    return transitions, branches


def rollout(world, case, prefix, model=None):
    state = prefix
    executed = []
    for step in range(BUDGET):
        remaining = BUDGET - step
        actions, features, mask = proposals(world, state, remaining)
        if model is None:
            values = rule_scores(world, state.belief, state.previous, actions, remaining)
        else:
            from learners import scores
            values = scores(model, features[None], mask[None])[0]
        action = actions[int(np.argmax(values))]
        if action is None:
            raise RuntimeError("policy returned illegal slot")
        state = apply(world, case, state, action)
        executed.append(action)
    return evaluate_outcome(world, case, prefix, state, executed, trace=True)


def collect(args, world):
    rows, reports = [], []
    for index in range(args.start, args.start + args.count):
        seed = seed_at("train", index)
        case = world.case(seed)
        prefix = prepared(world, case)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 7001]))
        transitions, branches = generate_one(world, case, prefix, rng=rng)
        rows.extend({**row, "case_index": index, "case_seed": seed} for row in transitions)
        reports.append({"index": index, "seed": seed, "branches": branches,
                        "prefix_acquisitions": len(prefix.frames)})
    fields = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    np.savez_compressed(args.out / f"train_{args.start:04d}.npz", **fields)
    (args.out / f"train_{args.start:04d}.json").write_text(json.dumps(reports, allow_nan=False))
    print(f"{args.count} cases, {len(rows)} transitions at index {args.start}", flush=True)


def train_all(args):
    import torch
    import learners
    torch.set_num_threads(1)
    data_shards = []
    shards = sorted(args.out.glob("train_*.npz"))
    for path in shards:
        with np.load(path) as data:
            data_shards.append({key: data[key] for key in (*TRANSITION_KEYS, "case_index", "case_seed")})
    if not data_shards:
        raise ValueError("no training shards")
    table = {key: np.concatenate([part[key] for part in data_shards])
             for key in data_shards[0]}
    cases = np.unique(table["case_index"])
    if not np.array_equal(cases, np.arange(max(TRAINING_CASES))):
        raise ValueError("training cases incomplete")
    expected = np.array([seed_at("train", i) for i in range(max(TRAINING_CASES))])
    if not np.array_equal(table["case_seed"], expected[table["case_index"]]):
        raise ValueError("training seeds do not match the frozen case indices")
    for size in TRAINING_CASES:
        use = table["case_index"] < size
        training = {key: table[key][use] for key in TRANSITION_KEYS}
        for method in METHODS:
            for seed in TRAINING_SEEDS:
                name = f"{method}_{size}_{seed}"
                model = learners.train(training, method, seed=seed, steps=args.steps, device=args.device)
                learners.save(model, args.out / f"{name}.pt")
                print(name, len(training["reward"]), str(model.device), flush=True)
    manifest = {
        "case_counts": list(TRAINING_CASES), "case_indices": cases.tolist(),
        "case_seeds": np.unique(table["case_seed"]).tolist(),
        "steps": args.steps, "train_seeds": TRAINING_SEEDS,
        "rows_by_size": {str(n): int(np.sum(table["case_index"] < n)) for n in TRAINING_CASES},
        "models": {p.name: sha(p) for p in sorted(args.out.glob("*.pt"))},
        "sources": {name: sha(HERE / name) for name in CORE_FILES},
        "geometry": sha(GEOMETRY),
        "data": {p.name: sha(p) for p in shards},
    }
    (args.out / "TRAIN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


def test(args, world):
    import torch
    import learners
    torch.set_num_threads(1)
    manifest = json.loads((args.out / "TRAIN_MANIFEST.json").read_text())
    if manifest["geometry"] != sha(GEOMETRY) or any(sha(HERE / name) != digest
                                                   for name, digest in manifest["sources"].items()):
        raise ValueError("frozen study source or atlas changed")
    models = {}
    for name, digest in manifest["models"].items():
        path = args.out / name
        if sha(path) != digest:
            raise ValueError(f"checkpoint changed: {name}")
        models[path.stem] = learners.load(path, device="cpu")
    train_seeds = set(manifest["case_seeds"])
    with (args.out / f"{args.split}_{args.start:04d}.jsonl").open("w") as stream:
        for index in range(args.start, args.start + args.count):
            seed = seed_at(args.split, index)
            if seed in train_seeds:
                raise ValueError("test/training overlap")
            case = world.case(seed)
            prefix = prepared(world, case)
            runs = {"rule": rollout(world, case, prefix)}
            runs.update({name: rollout(world, case, prefix, model) for name, model in models.items()})
            prefix_ids = {row["prefix_sha256"] for row in runs.values()}
            if len(prefix_ids) != 1 or any(row["captures"] != BUDGET for row in runs.values()):
                raise ValueError("paired prefix or budget differs")
            stream.write(json.dumps({"index": index, "seed": seed, "runs": runs,
                                     "prefix_acquisitions": len(prefix.frames)}, allow_nan=False) + "\n")
            stream.flush()
        print(args.split, args.start, args.count, "cases", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collect", "train", "test"))
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "train":
        train_all(args)
    elif args.mode == "collect":
        collect(args, World())
    else:
        test(args, World())


if __name__ == "__main__":
    main()
