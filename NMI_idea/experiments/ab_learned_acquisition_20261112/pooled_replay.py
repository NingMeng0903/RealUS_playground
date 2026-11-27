"""Compare the fitted observation models on the same saved acquisition decisions."""

import json
from pathlib import Path

import numpy as np

import acquisition as trial


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "acquisition"


def main():
    train = json.loads((OUTPUT / "TRAIN.json").read_text())
    rows = [json.loads(line) for line in (OUTPUT / "TEST.jsonl").read_text().splitlines()]
    world = trial.World()
    schedule = trial.build_schedule(world)
    with np.load(world.geometry_path, allow_pickle=False) as asset:
        trial.VIEW_TILTS = asset["view_tilts_deg"]
    pooled = {int(view): trial.Clutter(**model) for view, model in train["fits"]["pooled"].items()}
    result = {}
    rng = np.random.default_rng(2026111216)
    for mode in trial.STRESS_CASES:
        nll = []
        for row in rows:
            if row["mode"] != mode:
                continue
            nominal = row["arms"]["nominal"]
            case = world.case(row["seed"])
            prefix = trial.first_sweep(world, case, mode)
            previous = prefix[-1]
            suffix = []
            for stage, view in enumerate(nominal["views"]):
                reading = trial.Reading(previous.t, previous.q, previous.velocity)
                action = next(a for a in schedule.options(world, reading, previous, stage) if a.view == view)
                frame = trial.acquire(world, case, action, mode)
                assert frame.t == nominal["times"][stage]
                assert frame.centroids.tolist() == nominal["frame_centroids"][stage]
                suffix.append((action, frame))
                previous = frame
            observed, _ = trial.run_arm(world, schedule, case, mode, prefix, pooled, suffix)
            nll.append((nominal["nll"], observed["nll"], row["arms"]["view_replay"]["nll"]))
        values = np.asarray(nll)
        indices = rng.integers(len(values), size=(trial.BOOTSTRAPS, len(values)))
        result[mode] = {"cases": len(values), "nll_means": dict(zip(
            ("nominal", "pooled", "view"), values.mean(axis=0).tolist(),
        ))}
        for left, right, key in ((1, 0, "pooled_minus_nominal"), (2, 1, "view_minus_pooled")):
            diff = values[:, left] - values[:, right]
            result[mode][key] = {
                "difference": float(diff.mean()),
                "ci95": np.quantile(diff[indices].mean(axis=1), [.025, .975]).tolist(),
            }
    (OUTPUT / "SAME_FRAME_POOLING.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
