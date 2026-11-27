"""Run the frozen response model with explicit old evidence and a fresh request."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "residual_scene"
sys.path.insert(0, str(SOURCE))
import scene

SEED, COUNT = 721229, 128
TARGET = "target_patch"
METHODS = ("address_cache", "full_replay", "window4", "direct_cache", "stale_cache", "fixed_pair")


@dataclass(frozen=True)
class Event:
    identifier: int
    address: str
    epoch: int
    action: int
    y: float


def replay(tree, events, epoch):
    natural, counts, seen = tree.natural.copy(), (0, 0), set()
    for e in events:
        if e.identifier in seen or e.address != TARGET or e.epoch != epoch:
            continue
        seen.add(e.identifier)
        natural, counts = tree.update(natural, counts, e.action, e.y)
    return natural, counts


def acquire(tree, case, state, method):
    natural, counts = state
    current, elapsed = scene.START, 0.
    seen = np.zeros(len(scene.PROBES), dtype=int)
    actions, observations = [], []
    for step in range(4):
        remaining = scene.BUDGET - elapsed
        if method == "fixed_pair" and step < len(scene.PROBES):
            action = scene.PROBES[step]
        else:
            depth = 1 if method in ("direct_cache", "fixed_pair") else 4 - step
            values = tree.options(natural, counts, current, remaining, depth)
            action = (scene.TERMINALS + scene.PROBES)[int(np.argmax(values))]
        elapsed += tree.cost[current, action]
        if elapsed > scene.BUDGET + 1e-10:
            raise AssertionError("revisit exceeds the shared budget")
        actions.append(action)
        if action in scene.TERMINALS:
            correct, physical, acoustic = case.terminal(action, False)
            return {"actions": actions, "public_new_probe_y": observations, "elapsed_s": float(elapsed),
                    "correct": correct, "physical": physical, "acoustic": acoustic}
        index = scene.PROBES.index(action)
        y = case.probe(action, seen[index])
        seen[index] += 1
        observations.append(y)
        natural, counts = tree.update(natural, counts, action, y)
        current = action
    raise AssertionError("revisit did not acquire a fresh terminal observation")


def run():
    model = scene.Model.from_file(SOURCE / "MODEL.json")
    duration = scene.motion.duration_matrix(scene.POSES)
    tree = scene.PublicTree(model, scene.geometry(scene.NOMINAL), duration)
    geometry, latent, _, rng = scene.sample_objects(SEED, COUNT)
    renewed = np.random.default_rng(np.random.SeedSequence([SEED, 1])).normal(size=latent.shape) * scene.TRUE_SD
    prep_noise = rng.normal(size=(COUNT, len(scene.PROBES))) * scene.SENSOR_SD
    new_noise = rng.normal(size=(COUNT, len(scene.PROBES), scene.MAX_PROBES)) * scene.SENSOR_SD
    terminal_noise = rng.normal(size=(COUNT, len(scene.TERMINALS))) * scene.SENSOR_SD
    first, second = scene.PROBES
    prep_cost = tree.cost[scene.START, first] + tree.cost[first, second] + duration[second, scene.START] + .8
    rows = {}
    for epoch, label in enumerate(("stable", "reset")):
        rows[label] = {method: [] for method in METHODS}
        for i in range(COUNT):
            old = geometry["base"][i, list(scene.PROBES)] + scene.FEATURES[list(scene.PROBES)] @ latent[i]
            old += prep_noise[i]
            common = [Event(j, TARGET, 0, a, float(y)) for j, (a, y) in enumerate(zip(scene.PROBES, old))]
            empty = (tree.natural.copy(), (0, 0))
            prior_snapshot = replay(tree, common, 0)
            full_ledger = common + [Event(2 + k, "different_patch", 0, -1, 0.) for k in range(4)]
            cache = prior_snapshot if epoch == 0 else empty
            state = {
                "address_cache": cache,
                "full_replay": replay(tree, full_ledger, epoch),
                "window4": replay(tree, full_ledger[-4:], epoch),
                "direct_cache": cache,
                "stale_cache": prior_snapshot,
                "fixed_pair": empty,
            }
            case = scene.PrivateCase(
                geometry["base"][i], geometry["gate"][i], latent[i] if epoch == 0 else renewed[i],
                0., new_noise[i], terminal_noise[i],
            )
            for method in METHODS:
                rows[label][method].append(acquire(tree, case, state[method], method))
        if rows[label]["address_cache"] != rows[label]["full_replay"]:
            raise AssertionError("cache and full replay disagree")
        if epoch == 1 and rows[label]["address_cache"] != rows[label]["window4"]:
            raise AssertionError("reset-aware cache should agree with fresh window")

    boot = np.random.default_rng(721230).integers(COUNT, size=(4000, COUNT))
    summaries, differences = {}, {}
    for arm, group in rows.items():
        hit = {m: np.array([r["correct"] for r in data], dtype=float) for m, data in group.items()}
        elapsed = {m: np.array([r["elapsed_s"] for r in data]) for m, data in group.items()}
        probes = {m: np.array([len(r["actions"]) - 1 for r in data]) for m, data in group.items()}
        summaries[arm] = {
            m: {
                "correct": int(hit[m].sum()), "fresh_mean_s": float(elapsed[m].mean()),
                "including_preparation_mean_s": float(elapsed[m].mean() + prep_cost),
                "fresh_probes_mean": float(probes[m].mean()),
                "fresh_direct_count": int((probes[m] == 0).sum()),
            }
            for m in METHODS
        }
        differences[arm] = {}
        for left, right in (("address_cache", "window4"), ("address_cache", "fixed_pair"),
                            ("address_cache", "direct_cache"), ("address_cache", "stale_cache")):
            outcome = {}
            for name, table in (("correct_rate", hit), ("fresh_elapsed_s", elapsed), ("fresh_probes", probes)):
                delta = table[left] - table[right]
                outcome[name] = {"mean": float(delta.mean()),
                                 "object_bootstrap_95": np.quantile(delta[boot].mean(axis=1), [.025, .975]).tolist()}
            differences[arm][f"{left}_minus_{right}"] = outcome
    files = [HERE / "PROTOCOL.txt", Path(__file__), SOURCE / "scene.py", SOURCE / "MODEL.json", SOURCE / "ROBOT.json"]
    return {"seed": SEED, "objects": COUNT, "shared_preparation_s": float(prep_cost),
            "summary": summaries, "paired": differences, "rows": rows,
            "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1, user_api="blas"):
        result = run()
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
