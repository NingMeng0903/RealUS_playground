"""Independent numerical checks of the finite-segment enclosure and mean control."""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import segment_v16 as rule


def checks():
    rng = np.random.default_rng(761609)
    discrepancy = -np.inf
    for _ in range(200):
        frame = Rotation.from_rotvec(rng.normal(0., .18, 3)).as_matrix() @ rule.base_frame(0)
        sweep = rule.Sweep(0, "analytic", frame)
        origin = rng.normal(0., 6., 3)
        centers = origin + rng.uniform([-85., -8., -18.], [-1., 8., 18.], (6, 3))
        tangents = rule.unit(rng.normal(0., .30, (6, 3)) + np.array([0., 1., 0.]))
        upper, _, _, _, ok = rule._checks(sweep, origin, centers, tangents)
        exact_frames = sweep.frame(np.linspace(0., 1., 2049))
        d = centers - origin
        error = rule.residual(exact_frames[:, None], d, tangents)
        discrepancy = max(discrepancy, float((np.linalg.norm(error, axis=-1).max(axis=0) - upper).max()))
        assert np.all(exact_frames[:, :, 2] @ d[ok].T >= 3.)
        assert np.all(exact_frames[:, :, 2] @ d[ok].T <= 80.)
        assert np.all(exact_frames[:, :, 1] @ tangents[ok].T > 0.)
    assert discrepancy < 0.

    neutral = rule.base_frame(0)
    centers = np.array([[-35., 0., 6.], [-35., 0., -6.]])
    tangents = np.tile([0., 1., 0.], (2, 1))
    old = rule.Belief(centers, tangents, np.array([.5, .5]), (), (.5, .5, 0.))
    after = rule.Belief(centers, tangents, np.array([.93, .07]), (), (.93, .07, 0.))
    origin = np.zeros(3)
    middle_sweep = rule.Sweep(0, "mid", neutral)
    target_a = rule.Sweep(0, "A", rule.aim(origin, neutral, centers[0], tangents[0]))
    old_mean = rule.evaluate(middle_sweep, origin, old, False)
    old_full = rule.evaluate(middle_sweep, origin, old, True)
    enhanced = rule.evaluate(target_a, origin, after, True)
    assert old_mean["certified"] and not old_full["certified"] and enhanced["certified"]
    assert rule.validate_segment(target_a, origin, centers[0], tangents[[0]])["pass"]
    assert not rule.validate_segment(target_a, origin, centers[1], tangents[[1]])["pass"]
    return {"seed": 761609, "random_sweeps": 200, "hypotheses_per_sweep": 6,
            "angles_per_sweep": 2049, "max_true_lower_minus_enclosure": discrepancy,
            "mean_control_certifies_between_two_possible_branches": old_mean["certified"],
            "finite_support_certifies_middle": old_full["certified"],
            "finite_support_certifies_target_after_0p93": enhanced["certified"],
            "independent_target_A_geometry_passed": True}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(checks(), indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(payload, end="")
