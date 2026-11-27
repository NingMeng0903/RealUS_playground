"""Isolated first-scan variant using frozen candidate-information features.

This wrapper leaves the compact first-scan pilot untouched and replaces only
the CQL feature builder with the frozen ``proposal_features`` hit/separation
builder.  Seeds, candidate paths, rewards, masks, done flags and update budget
are inherited unchanged from ``first_scan.py`` and checked after collection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import first_scan as base  # noqa: E402


def matched_fork_features(world, state, site, schedule, stage):
    actions = base.fork_actions(world, state, site, schedule)
    features, mask = base.proposal_features(
        world, state.belief, actions, len(base.FORK_SITES) - stage, len(base.FORK_SITES),
    )
    return actions, features, mask


def parity(compact: Path, matched: Path) -> dict:
    with np.load(compact / "TRAIN_TRANSITIONS.npz") as a, np.load(matched / "TRAIN_TRANSITIONS.npz") as b:
        fields = ("mask", "action", "reward", "next_mask", "done")
        checks = {field: bool(np.array_equal(a[field], b[field])) for field in fields}
        checks["row_count_equal"] = bool(a["action"].shape == b["action"].shape)
        checks["feature_shapes"] = {"compact": list(a["features"].shape),
                                    "matched": list(b["features"].shape)}
        checks["features_differ"] = bool(not np.array_equal(a["features"], b["features"]))
    compact_cases = json.loads((compact / "TRAIN_CASES.json").read_text())
    matched_cases = json.loads((matched / "TRAIN_CASES.json").read_text())
    checks["case_seed_and_paths_equal"] = bool(compact_cases == matched_cases)
    checks["all_transition_parity"] = bool(all(checks[field] for field in
                                                ("mask", "action", "reward", "next_mask", "done"))
                                             and checks["case_seed_and_paths_equal"])
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "first_scan_matched_features")
    parser.add_argument("--compact-out", type=Path, default=HERE / "first_scan")
    parser.add_argument("--train-count", type=int, default=128)
    parser.add_argument("--test-start", type=int, default=512)
    parser.add_argument("--test-count", type=int, default=128)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    base.fork_features = matched_fork_features
    world = base.World()
    table = base.collect_training(world, count=args.train_count, out=args.out)
    model_paths = []
    import torch
    torch.set_num_threads(1)
    for seed in base.TRAIN_SEEDS:
        path = args.out / f"cql_firstscan_matched_{seed}.pt"
        model = base.learners.train(table, "cql", seed=seed, steps=args.steps,
                                    device=args.device, hidden=64, depth=2,
                                    batch_size=256, cql_alpha=.50)
        base.learners.save(model, path)
        model_paths.append(path)
    base.write_json(args.out / "TRAIN_MANIFEST.json", {
        "cases": args.train_count, "rows": int(len(table["reward"])),
        "seeds": list(base.TRAIN_SEEDS), "steps": args.steps,
        "feature_builder": "frozen proposal_features (candidate hit/separation)",
        "models": {p.name: base.sha(p) for p in model_paths},
    })
    # Test/summarize use the same base functions and output schema.
    class TestArgs:
        pass
    test_args = TestArgs()
    test_args.out = args.out
    test_args.test_start = args.test_start
    test_args.test_count = args.test_count
    base.run_test(test_args, world, model_paths)
    summary = base.summarize(test_args, world)
    base.plot_3d(test_args, world, summary)
    checks = parity(args.compact_out, args.out)
    base.write_json(args.out / "PARITY.json", checks)
    base.write_json(args.out / "MANIFEST.json", {
        "variant": "matched_candidate_information_features",
        "base_driver": base.sha(HERE / "first_scan.py"),
        "wrapper": base.sha(HERE / "first_scan_matched_features.py"),
        "compact_root": str(args.compact_out), "models": {p.name: base.sha(p) for p in model_paths},
        "config": {"train_count": args.train_count, "test_start": args.test_start,
                   "test_count": args.test_count, "steps": args.steps,
                   "seeds": list(base.TRAIN_SEEDS)},
        "parity": checks,
        "no_recheck_checkpoint_reused": True,
    })
    print(json.dumps({"out": str(args.out), "parity": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
