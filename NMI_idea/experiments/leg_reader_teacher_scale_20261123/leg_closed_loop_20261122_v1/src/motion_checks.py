"""Reproduce the geometry, causal feedback, and matched-protocol checks."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

import motion


class Relabeled(motion.Trial):
    """Alter the evaluator channel while preserving every public observation."""

    def observe(self, *args, **kwargs):
        echoes, labels, quality = super().observe(*args, **kwargs)
        return echoes, (labels + 1) % 6, quality


class ShiftedEchoes(motion.Trial):
    """Perturb only the anonymous returns observed after the shared initial state."""

    def observe(self, points, gates, tick, **kwargs):
        echoes, labels, quality = super().observe(points, gates, tick, **kwargs)
        echoes[:, 0] = np.clip(echoes[:, 0] + 4., -25., 25.)
        return echoes, labels, quality


def legacy_anchor_check(scanner, legacy_file):
    from student_transfer import move, start

    env = motion.legacy_scan
    env.STORE = env.Store(legacy_file.parent)
    old_model = env.Model(30., True, True)
    results = {}
    for name, spec in motion.holdout_specs(scanner).items():
        trial = motion.Trial(3, 884001, "ordinary")
        old = start(spec, trial, old_model)
        new = scanner.start(spec, trial, scanner.model(30.))
        frames = 0
        while True:
            assert (old.station, old.sector, old.tick) == (new.station, new.sector, new.tick)
            np.testing.assert_allclose(old.distance, new.distance, rtol=1.e-14)
            old_row, new_row = old.reader.frames[-1], new.reader.frames[-1]
            for i in (2, 3, 4, 6, 7):
                np.testing.assert_allclose(old_row[i], new_row[i], rtol=1.e-12, atol=1.e-13)
            frames += 1
            if old.station == spec[2]:
                break
            following = old.station + 1
            a = env.pick_angle(old_model, old.reader.w, old.station, old.sector, following)
            move(old, following, a, trial, spec)
            scanner.advance(new, spec, trial)
        results[name] = {"matched_anchors": frames, "ticks": old.tick, "distance_mm": old.distance}
    return results


def motion_check(scanner, spec, source):
    action = motion.ContinuousAction(.25, -.25, 0.)
    options = {"max_ticks": 32, "max_mm": 85.}
    result = {}
    for profile in ("ordinary", "stress"):
        trial = motion.Trial(source, 331, profile)
        relabeled = Relabeled(source, 331, profile)
        node = scanner.start(spec, trial, scanner.model(30.))
        other = scanner.start(spec, relabeled, scanner.model(30.))
        for _ in range(4):
            scanner.advance(node, spec, trial)
            scanner.advance(other, spec, relabeled)
        time_before, frames_before = node.tick, len(node.reader.frames)
        np.testing.assert_array_equal(motion.public_features(node, spec), motion.public_features(other, spec))
        opened = scanner.prodmp(node, spec, trial, action, feedback=False, **options)
        closed = scanner.prodmp(node, spec, trial, action, **options)
        permuted = scanner.prodmp(other, spec, relabeled, action, **options)
        shifted = scanner.prodmp(node, spec, ShiftedEchoes(source, 331, profile), action, **options)
        shifted_opened = scanner.prodmp(node, spec, ShiftedEchoes(source, 331, profile),
                                        action, feedback=False, **options)
        discrete = scanner.mpc(node, spec, trial, action, **options)
        assert all(item.legal for item in (opened, closed, permuted, shifted, shifted_opened, discrete))
        np.testing.assert_array_equal(closed.trajectory, permuted.trajectory)
        np.testing.assert_array_equal(opened.trajectory, shifted_opened.trajectory)
        for item in (opened, closed, permuted, shifted, shifted_opened, discrete):
            assert item.ticks <= options["max_ticks"] and item.distance <= options["max_mm"]
            assert item.node.station == node.station and item.node.sector == node.sector
            assert len(item.node.reader.frames) - frames_before == item.ticks
            assert item.trajectory[:, 3].max() <= motion.MAX_SPEED_MM_S * motion.CLOCK_S + 1.e-7
            np.testing.assert_array_equal(item.trajectory[:, 0], np.arange(time_before + 1, item.node.tick + 1))
        rejected = scanner.prodmp(node, spec, trial, action, max_ticks=0, max_mm=0.)
        assert not rejected.legal and rejected.node is node
        assert rejected.ticks == 0 and rejected.distance == 0.
        assert node.tick == time_before and len(node.reader.frames) == frames_before
        assert np.any(np.abs(closed.trajectory[:, 1:3] - np.rint(closed.trajectory[:, 1:3])) > 1.e-5)
        result[profile] = {
            "ticks": closed.ticks,
            "distance_mm": closed.distance,
            "feedback_updates": closed.feedback_updates,
            "max_feedback_shift_sector": float(np.max(np.abs(closed.trajectory[:, 2] - opened.trajectory[:, 2]))),
            "max_echo_perturbation_shift_sector": float(np.max(np.abs(closed.trajectory[:, 2] - shifted.trajectory[:, 2]))),
            "mpc_probe_ticks": discrete.ticks,
            "mpc_probe_distance_mm": discrete.distance,
        }
    assert max(row["max_feedback_shift_sector"] for row in result.values()) > .015
    assert max(row["max_echo_perturbation_shift_sector"] for row in result.values()) > .015
    return result


def shape_check():
    initial, velocity, final = np.array([7., 4.]), np.array([.01, .015]), np.array([6., 5.])
    paths = [motion.PRIMITIVE.segment(initial, velocity, final, 10, np.array([0., shape]))
             for shape in (-.4, .4)]
    for positions, velocities in paths:
        np.testing.assert_array_equal(positions[[0, -1]], [initial, final])
        np.testing.assert_array_equal(velocities[[0, -1]], [velocity, [0., 0.]])
    return float(np.max(np.abs(paths[0][0][:, 1] - paths[1][0][:, 1])))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--ct-pack", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    ct = motion.Scanner.training(args.ct_pack)
    if ct._anatomy.links is None:
        raise ValueError("The check requires linked training topology")
    shape_shift = shape_check()
    assert shape_shift > .5
    result = {
        "archived_basis": motion.BASIS_SOURCE,
        "opposite_shape_max_shift_sector_shared_endpoints": shape_shift,
        "medseg_span_neutral_mm": [float(ct.stations[i]) for i in ct.ct_bounds],
        "ct_motion": motion_check(ct, motion.definition(0, *ct.ct_bounds), 3),
        "verified": [
            "label-channel intervention left public features and trajectories identical",
            "echo perturbation shifted the closed-loop trajectory and preserved the open-loop trajectory",
            "sub-grid acquisition, route-pose recovery, one echo packet per tick and hard physical bounds",
            "zero-budget probe left the original state and frame archive intact",
        ],
    }
    report = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out is not None:
        args.out.write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
