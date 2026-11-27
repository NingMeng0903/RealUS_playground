"""Evaluator-only oracle coverage diagnostic for fixed public pose lattices."""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

from run_stress import _shifted_worlds
from sensor import env


HERE = Path(__file__).resolve().parent


def evaluate(worlds, poses):
    """Chunk the renderer; keep the declared task gates exactly fixed."""
    angles, centers, lengths = [], [], []
    for start in range(0, len(poses), 32):
        prediction = env.predict_batch(worlds, poses[start:start + 32])
        lengths.append(prediction["geometric_length"][..., 0])
        centers.append(np.abs(prediction["means"][..., 0, 0]))
        angles.append(prediction["angle_audit"][..., 0])
    length = np.concatenate(lengths, axis=1)
    center = np.concatenate(centers, axis=1)
    angle = np.concatenate(angles, axis=1)
    visible = length > .0005
    long_enough = length >= .015
    oriented = angle <= np.deg2rad(10)
    centered = center <= .002
    qualified = long_enough & oriented & centered
    return {
        "pose_count": len(poses),
        "objects_with_visible_pose": int(visible.any(axis=1).sum()),
        "objects_with_long_pose": int(long_enough.any(axis=1).sum()),
        "objects_with_long_and_oriented_pose": int((long_enough & oriented).any(axis=1).sum()),
        "objects_with_long_and_centered_pose": int((long_enough & centered).any(axis=1).sum()),
        "objects_with_qualified_pose": int(qualified.any(axis=1).sum()),
        "qualified_pose_count_by_object": qualified.sum(axis=1).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads((HERE / "PROTOCOL.json").read_text())["geometry"]
    cap = json.loads((HERE / "CAPABILITY_STRESS.json").read_text())
    seed, count = spec["test_seed"], spec["test_objects"]
    worlds = _shifted_worlds(seed, count, spec)
    coarse = np.asarray(cap["poses"])
    lattices = {"coarse_31": coarse}
    for step in (.004, .002, .001):
        lateral = np.arange(-.032, .03201, step)
        yaw = np.arange(-.68, .681, .17)
        dense = np.asarray(list(product((0.0,), lateral, yaw)))
        name = f"lateral_step_{1000 * step:g}mm_yaw_step_0.17rad"
        lattices[name] = np.unique(np.round(np.vstack((coarse, dense)), 9), axis=0)
    # The extra pose sets are only inspected by the evaluator. IK and any
    # controller or policy outcome are intentionally outside this diagnostic.
    result = {
        "scope": "Post-run oracle coverage on the existing held-out geometry, excluding new IK and never feeding results to a policy. All six action/observation budgets and task gates are unchanged in the original study.",
        "seed": seed,
        "independent_objects": count,
        "lattices": {name: evaluate(worlds, poses) for name, poses in lattices.items()},
    }
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
