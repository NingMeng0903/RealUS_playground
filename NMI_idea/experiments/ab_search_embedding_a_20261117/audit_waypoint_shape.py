"""Compare recorded CEM shapes with equal-endpoint references on fresh disturbances."""

import argparse
import gzip
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from run_adaptive_recovery import (WORKERS, init_worker, jsonable, paired_statistics, real_trial,
                                   shared_prefix, tape_for)


FIELDS = ("corrected_complete", "fresh_query_confirmed", "first_block_true_image", "new_centered_mistake",
          "final_wrong", "tcp_mm", "turn_degs", "seconds", "frames")
ALL_ARMS = ("endpoint_only", "cem_waypoint", "prodmp", "mpc")


def run_case(job):
    row, count = job
    family, seed = row["family"], row["seed"]
    controller = WORKERS[family]
    scene = controller.world.draw(seed)
    root, prefix = shared_prefix(controller, scene)
    assert root.frame == row["checkpoint"]["frame"]
    np.testing.assert_allclose(root.belief, row["public_root"]["belief"], atol=1e-12, rtol=0.)
    request = row["example"]["request"]
    ds, dx, bow = request
    actions = {"endpoint_only": controller.free_candidate(root, "cem", ds, dx, 0.),
               "cem_waypoint": controller.free_candidate(root, "cem", ds, dx, bow),
               "prodmp": controller.free_candidate(root, "cem", ds, dx, bow, renderer="fit"),
               "mpc": controller.choose_mpc(root, expanded=True, open_decision=True)}
    reference = actions["cem_waypoint"]
    assert reference is not None
    np.testing.assert_allclose(np.column_stack((reference.lateral.x, reference.lateral.v)),
                               row["example"]["nominal_x_vx"], atol=1e-12, rtol=0.)
    for arm in ("endpoint_only", "prodmp"):
        other = actions[arm]
        if other is not None:
            np.testing.assert_array_equal(other.s, reference.s)
            np.testing.assert_array_equal(other.vs, reference.vs)
            np.testing.assert_allclose(other.lateral.x[[0, -1]], reference.lateral.x[[0, -1]], atol=1e-12, rtol=0.)
            np.testing.assert_allclose(other.lateral.v[[0, -1]], reference.lateral.v[[0, -1]], atol=1e-12, rtol=0.)

    results = []
    for replica in range(count):
        # Disjoint namespace from the CEM search, its admission replay and all policy studies.
        tape = tape_for(scene.latent, seed, replica, 14271)
        outcomes = {}
        for arm, action in actions.items():
            if action is not None:
                metrics, record = real_trial(controller, root, prefix, tape, [action])
                actual = record.decisions[len(prefix.decisions)]
                outcomes[arm] = {**{key: metrics[key] for key in FIELDS},
                                 "first_executed_goal": actual["executed_goal"],
                                 "first_tcp_mm": actual["executed_tcp_mm"]}
        results.append(outcomes)
    return {"family": family, "seed": seed, "request": request, "root_type": row["checkpoint"]["category"],
            "first_nominal_goal": [float(reference.s[-1]), reference.lateral.goal],
            "feasible": {arm: action is not None for arm, action in actions.items()}, "replicas": results}


def summary(rows):
    tests = {"all": rows, "ordinary": [r for r in rows if r["family"] in ("connection", "parallel")]}
    report = {"new_tape_purpose": 14271, "selected_development_worlds": len(rows),
              "feasible_worlds": {a: sum(row["feasible"][a] for row in rows) for a in ALL_ARMS}}
    for name, subset in tests.items():
        matched = [r for r in subset if r["feasible"]["endpoint_only"] and r["feasible"]["mpc"]]
        if matched:
            report[name] = paired_statistics(matched, ("endpoint_only", "cem_waypoint", "mpc"),
                                             FIELDS, lambda r: r["replicas"])
            nominal_deltas = [np.mean([t["cem_waypoint"]["corrected_complete"]
                                       - t["endpoint_only"]["corrected_complete"] for t in row["replicas"]])
                              for row in matched]
            report[name]["worlds_waypoint_better_worse_tied"] = [sum(x > 0. for x in nominal_deltas),
                                                               sum(x < 0. for x in nominal_deltas),
                                                               sum(x == 0. for x in nominal_deltas)]
    fitted = [r for r in rows if r["feasible"]["prodmp"]]
    if fitted:
        report["shape_representation"] = paired_statistics(fitted, ("cem_waypoint", "prodmp"), FIELDS,
                                                           lambda r: r["replicas"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--replicas", type=int, default=12)
    args = parser.parse_args()
    if min(args.jobs, args.replicas) < 1:
        parser.error("jobs and replicas must be positive")
    with gzip.open(args.archive, "rt") as stream:
        selected = [r for r in map(json.loads, stream) if r["example"] is not None]
    if not selected:
        parser.error("the archive contains no admitted nominal examples")
    init_worker(np.load(args.fit))
    with get_context("fork").Pool(args.jobs) as pool:
        rows = list(pool.imap(run_case, ((r, args.replicas) for r in selected)))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary(rows), default=jsonable, indent=2) + "\n")
    with gzip.open(args.out / "cases.jsonl.gz", "wt", compresslevel=3) as stream:
        for row in rows:
            stream.write(json.dumps(row, default=jsonable, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
