"""Verify the atlas axis and the label-free execution boundary on actual prefixes."""

from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

from study_worlds import DEFAULT_PACK, ROOT, atomic_json, cases_for, init_worlds, read_prefix, start
import numpy as np
from active_trial import weighted_step
from public_features import CorrectedPublicHistoryReader, public_arrays, replay_reader
from scale_worlds import VectorHistoryReader


def axis_check():
    class Coordinates:
        def absolute(self, station, sector, xy):
            return xy

    class Corrected(Coordinates, CorrectedPublicHistoryReader):
        pass

    class Archived(Coordinates, VectorHistoryReader):
        pass

    priors = np.array([[[[0., 0.], [20., 0.]]], [[[20., 80.], [20., 0.]]]])
    model = SimpleNamespace(prior=np.asarray([.5, .5]), surface=SimpleNamespace(y=np.asarray([0.]), priors=priors))
    frame = (0., 0., np.array([[0., 0.], [20., 0.]]), None, .8, 0, np.full(2, .5), np.full(2, .2), 1)
    new = Corrected(model, (0, 1), True)
    old = Archived(model, (0, 1), True)
    _, previous = old._new_geometry(frame)
    _, score = new._new_geometry(frame)
    model.surface.priors = model.surface.priors[::-1]
    _, permuted = new._new_geometry(frame)
    np.testing.assert_array_equal(score, permuted)
    assert score[0] > score[1]
    return {"old": previous.tolist(), "corrected": score.tolist(), "source_swap_invariant": True}


def observed_checks(pack):
    init_worlds(pack)
    total = []
    for case in cases_for("smoke", 8):
        env, spec, node, trial, limits = start(case)
        for _ in range(6):
            env.route.advance(node, spec, trial, limits)
        original = read_prefix(env, spec, node)
        frames = [(*f[:3], np.full_like(f[3], 5), *f[4:]) for f in node.reader.frames]
        replay = replay_reader(node.reader, frames, node.reader.mixed_posterior)
        other = public_arrays(replay, spec, env.material)
        for key in ("features", "frames", "slots", "points"):
            np.testing.assert_array_equal(original[key], other[key])
        left, right = node.fork(), node.fork()
        env.route.advance(left, spec, trial, limits)
        weighted_step(env, spec, right, trial, limits, right.reader.w.copy())
        assert (left.station, left.sector, left.distance, left.tick) == (right.station, right.sector, right.distance, right.tick)
        for a, b in zip(left.reader.frames, right.reader.frames, strict=True):
            np.testing.assert_array_equal(a[2], b[2])
            np.testing.assert_array_equal(a[6], b[6])
        total.append({"route": case.task, "labels_invariant": True, "unchanged_weights_preserve_MPC": True})
    return total


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-pack", default=DEFAULT_PACK, type=Path)
    parser.add_argument("--out", default=ROOT / "smoke_v4" / "contracts.json", type=Path)
    args = parser.parse_args()
    outcome = {"axis": axis_check(), "checks": observed_checks(args.atlas_pack)}
    atomic_json(args.out, outcome)
    print(outcome["axis"], len(outcome["checks"]), "actual poses checked", flush=True)


if __name__ == "__main__":
    main()
