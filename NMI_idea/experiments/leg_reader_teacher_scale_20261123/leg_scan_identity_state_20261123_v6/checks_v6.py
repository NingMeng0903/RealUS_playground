"""Exercise v6 prefix invariance and v5 executable comparator parity."""

from argparse import ArgumentParser
from pathlib import Path

from v6_data import DEFAULT_PACK, ROOT, atomic_json, cases, read_json
from anchor_experiment import execute as archived_execute
from anchor_policy import Session as ArchivedReader
from causal_reader import PrefixReader, model_state
from live_programs import execute, init_live, prefix_setup
from public_features import replay_reader
from reader_study import frozen_reader, warm_bundle
from study_data import complete, scene_links
import numpy as np


def rebuild(node, count):
    copied = node.fork()
    copied.reader = replay_reader(node.reader, node.reader.frames[:count], node.reader.mixed_posterior[:count])
    copied.station, copied.sector, copied.tick = copied.reader.frames[-1][0], copied.reader.frames[-1][1], copied.reader.frames[-1][8]
    return copied


def compare_prefixes(ctx):
    env, spec, tape, limits = (ctx[k] for k in ("env", "spec", "tape", "limits"))
    last = ctx["node"].fork(); complete(env, spec, last, tape, limits, ctx["anchor"].station)
    cfg = ctx["reader"]; count = len(last.reader.frames)
    begin = max(1, count - 9)
    schedules = (list(range(begin, count + 1)), list(range(begin, count, 3)) + [count], [begin, count], [count])
    data = []
    for schedule in schedules:
        reader = PrefixReader(env, spec, cfg)
        for boundary in schedule:
            node = rebuild(last, boundary)
            public, states = reader.update(node)
            assert reader.update(node)[1] is states
        data.append((public, states, reader.confidence(node)))
    reference, graph, conf = data[0]
    for public, states, score in data[1:]:
        for k in reference:
            np.testing.assert_array_equal(reference[k], public[k])
        for k in ("role", "object_role", "usable", "track_ids", "edges"):
            np.testing.assert_array_equal(graph["objects"][k], states["objects"][k])
        np.testing.assert_array_equal(conf, score)
    archived = model_state(reference, cfg["model"], scene_links(env), cfg["edge_model"], cfg["usability"])
    np.testing.assert_array_equal(archived["objects"]["object_role"], graph["objects"]["object_role"])

    legacy = []
    for schedule in (schedules[0], schedules[-1]):
        observer = ArchivedReader(env, spec, cfg["v5"])
        for boundary in schedule:
            node = rebuild(last, boundary)
            _, state = observer.update(node)
        legacy.append(state["unary"][:, 0])
    old_change = abs(legacy[0] - legacy[1])
    distorted = rebuild(last, count)
    distorted.reader.frames = [(*f[:3], np.full(len(f[2]), 42), *f[4:]) for f in distorted.reader.frames]
    _, fake = PrefixReader(env, spec, cfg).update(distorted)
    np.testing.assert_array_equal(fake["objects"]["role"], graph["objects"]["role"])
    return {"prefix_partition_points": schedules, "candidates": len(conf), "v6_max_difference": 0.,
            "v5_entry_cadence_target_delta_max": float(old_change.max(initial=0.)),
            "v5_entry_cadence_moved_more_than_0.01": int((old_change > .01).sum()), "swapped_label_invariant": True}


def compare_programs(ctx):
    v5 = archived_execute(ctx["env"], ctx["spec"], ctx["node"], ctx["tape"], ctx["limits"], ctx["anchor"],
                          ctx["clock"], ctx["v5_origin"], ctx["reader"]["v5"], "anchor_information")
    v6 = execute(ctx, ctx["tape"], "v5_public")
    compare = [[step["station"], step["sector"], step["tick"], step["distance_step_mm"]] for step in v5["forward_path"]]
    np.testing.assert_allclose(v6["forward_path"], compare, rtol=0., atol=1.e-9)
    np.testing.assert_allclose([v5["time_s"], v5["travel_mm"]], [v6["time_s"], v6["travel_mm"]], rtol=0., atol=1.e-8)
    for arm in ("prefix_frozen", "unary_dynamic", "objects_dynamic", "objects_no_stop", "objects_stop"):
        result = execute(ctx, ctx["tape"], arm)
        assert result["completed_legal"]
    return {"v5_forward_frames": len(compare), "v5_time_s": v6["time_s"], "v5_travel_mm": v6["travel_mm"],
            "still_legal": ["prefix_frozen", "unary_dynamic", "objects_dynamic", "objects_no_stop", "objects_stop"]}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=ROOT / "run", type=Path)
    parser.add_argument("--atlas-pack", default=DEFAULT_PACK, type=Path)
    args = parser.parse_args()
    base = warm_bundle(args.out, args.atlas_pack)
    data = read_json(args.out / "reader_freeze.json")
    init_live(args.atlas_pack, frozen_reader(base, data))
    checks = {}
    for case in list(cases("closed_dev"))[:2]:
        ctx = prefix_setup(case)
        checks[case.task] = compare_prefixes(ctx) | compare_programs(ctx)
        print(case.task, checks[case.task]["v5_entry_cadence_target_delta_max"], flush=True)
    atomic_json(args.out / "contracts.json", checks)


if __name__ == "__main__":
    main()
