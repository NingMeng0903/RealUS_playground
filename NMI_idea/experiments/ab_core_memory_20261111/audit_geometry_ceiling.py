"""Evaluate the geometry-only ceiling in the frozen local residual simulator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import ndtr


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "residual_scene"))
import scene


def run():
    report = scene.read(HERE / "residual_scene/TEST.json")
    n = report["n"]
    geometry, latent, _, noise_rng = scene.sample_objects(report["seed"], n)
    noise_rng.normal(size=(n, len(scene.PROBES), scene.MAX_PROBES))
    terminal_noise = noise_rng.normal(size=(n, len(scene.TERMINALS))) * scene.SENSOR_SD
    terminals = list(scene.TERMINALS)
    H = scene.FEATURES[terminals]
    # The simulator draws the residual and geometry independently. Even
    # with the true geometry, a policy insensitive to response observations
    # sees the same marginal Gaussian at each terminal.
    sd = np.sqrt((H**2 * scene.TRUE_SD**2).sum(axis=1) + scene.SENSOR_SD**2)
    base = geometry["base"][:, terminals]
    gate = geometry["gate"][:, terminals]
    probabilities = gate * ndtr(base / sd)
    choices = probabilities.argmax(axis=1)
    geom_expected = probabilities.max(axis=1)
    truth = gate & (base + latent @ H.T + terminal_noise >= 0)
    geom_paired = truth[np.arange(n), choices]
    original = report["rows"]["persistent"]["geometry"]
    original_choices = np.array([terminals.index(row["actions"][-1]) for row in original])
    if not np.array_equal(truth[np.arange(n), original_choices], [r["pass_"] for r in original]):
        raise ValueError("could not reproduce original geometry outcomes with the independent evaluator")

    boot = np.random.default_rng(721222).integers(n, size=(12000, n))
    comparisons = {}
    for method in ("h1", "h2", "h3", "h4", "fixed_pair_learned", "paired_difference"):
        passed = np.array([r["pass_"] for r in report["rows"]["persistent"][method]], dtype=bool)
        d = passed.astype(float) - geom_paired
        comparisons[method] = {
            "correct_count": int(passed.sum()),
            "paired_minus_oracle_count": int(d.sum()),
            "paired_delta_bootstrap_95": np.quantile(d[boot].mean(axis=1), [.025, .975]).tolist(),
        }
    return {
        "scope": (
            "Post hoc evaluator only, frozen synthetic response scene. The analytic ceiling is for "
            "the declared at-most-one terminal and geometry-only evidence class, including true "
            "local geometry in that class. Policies that use lawful response cues are outside it."
        ),
        "independent_objects": n,
        "privileged_geometry_expected_success_on_test_geometries": float(geom_expected.mean()),
        "geometry_expectation_95_across_geometries": np.quantile(
            geom_expected[boot].mean(axis=1), [.025, .975]
        ).tolist(),
        "privileged_geometry_empirical_correct_count": int(geom_paired.sum()),
        "privileged_geometry_action_counts": [int((choices == k).sum()) for k in range(len(terminals))],
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as out:
        json.dump(run(), out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
