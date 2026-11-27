"""Replay identical closed-loop observations with and without the knee lag state."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from belief import Filter, Reading
from knee_world import World
from summarize import collect_rows


def replay_one(world, record):
    case = world.case(record["seed"])
    run = record["runs"]["rule"]
    trace = run["trace"]
    velocities = []
    for frame in trace:
        regenerated = world.observe(case, frame["t"], frame["site"], frame["view"])
        assert regenerated.q == frame["q"]
        assert np.array_equal(regenerated.centroids, np.asarray(frame["centroids_mm"]).reshape(-1, 2))
        velocities.append(regenerated.velocity)
    values = {}
    for name, use_z in (("angle_and_speed", False), ("latent", True)):
        belief = Filter(latent=use_z)
        for event, (frame, velocity) in enumerate(zip(trace, velocities)):
            reading = Reading(frame["t"], frame["q"], velocity)
            belief.observe(world, reading, frame["site"], frame["view"], frame["centroids_mm"],
                           event=event, retain=False)
        squared, coverage = [], []
        for horizon in (.5, 1.):
            t = reading.t + horizon
            mean, sd = belief.forecast(world, t, world.query_sites)
            error = mean - world.truth_world(case, t, world.query_sites)
            squared.append(float(np.mean(np.sum(error**2, axis=1))))
            coverage.append(float(np.mean(np.abs(error) <= 1.6448536269514722 * sd)))
        if use_z:
            assert np.allclose(np.sqrt(squared), [r["rmse_mm"] for r in run["future"]], atol=1.e-9)
        values[name] = {"mse": squared, "coverage": coverage}
    return values


def main():
    path = Path(__file__).resolve().parent / "results"
    world = World()
    rows = [replay_one(world, r) for r in collect_rows(path, "test")]
    rng = np.random.default_rng(84062)
    resampling = rng.integers(0, len(rows), size=(4000, len(rows)))
    baseline = np.asarray([r["angle_and_speed"]["mse"] for r in rows])
    dynamic = np.asarray([r["latent"]["mse"] for r in rows])
    delta = np.sqrt(dynamic[resampling].mean(axis=1)) - np.sqrt(baseline[resampling].mean(axis=1))
    out = {
        "cases": len(rows), "horizons_s": [.5, 1.],
        "angle_and_speed_rmse_mm": np.sqrt(baseline.mean(axis=0)).tolist(),
        "latent_rmse_mm": np.sqrt(dynamic.mean(axis=0)).tolist(),
        "latent_minus_baseline_ci95_mm": np.quantile(delta, [.025, .975], axis=0).T.tolist(),
        "coverage": {key: np.mean([r[key]["coverage"] for r in rows], axis=0).tolist()
                     for key in rows[0]},
        "scope": "Identical lookahead-rule trace. Lag filter predictions reproduced from saved public observations.",
    }
    (path / "LATENT_REPLAY.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
