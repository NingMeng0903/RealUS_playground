"""Replay the same B-selected endpoint with the pre-CEM and CEM-fitted ProDMP."""

import argparse
from dataclasses import replace
import gzip
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from adaptive_controller import BLOCK_STEPS, MAX_TRAVEL, FreeCandidate, admissible
from run_adaptive_recovery import (WORKERS, init_worker, jsonable, paired_statistics, real_trial,
                                   shared_prefix, tape_for)


ARMS = ("reference_waypoint", "pre_cem_prodmp", "pre_cem_with_zero", "cem_fitted_prodmp")
FIELDS = ("corrected_complete", "final_wrong", "new_centered_mistake", "fresh_query_confirmed",
          "tcp_mm", "turn_degs", "seconds", "frames")


def legacy_reference(controller, state, action, weights):
    end = action.lateral.goal
    plan = controller.primitive.condition(0, BLOCK_STEPS, state.x, state.vx, end, 0., weights)
    plan.x[[0, -1]], plan.v[[0, -1]] = [state.x, end], [state.vx, 0.]
    if not admissible(plan):
        return None
    distance, turn = controller.geometry.cost(action.s, plan.x)
    if state.travel + distance > MAX_TRAVEL:
        return None
    return replace(action, lateral=plan, distance=distance, turn=turn)


def replay(job):
    study, tape_namespace, archived_arm, row = job
    family, seed = row["family"], row["seed"]
    controller = WORKERS[family]
    scene = controller.world.draw(seed)
    root, prefix = shared_prefix(controller, scene)
    assert root.frame == row["checkpoint"]["frame"]
    np.testing.assert_allclose(root.belief, row["public_root"]["belief"], atol=1e-12, rtol=0.)
    chosen = row["first_decision"].get("picked") if study == "primary" else row["selected"][archived_arm]
    reference = controller.options_shared(root).get(chosen)
    if not isinstance(reference, FreeCandidate):
        return None
    # The former continuous search used this shared side-scan skill in either
    # direction. Its native second choice uses zero shape residual at the same goal.
    weights = controller.shape("left", root.x, root.vx, reference.lateral.goal)
    original = legacy_reference(controller, root, reference, weights)
    proposals = {
        "reference_waypoint": reference,
        "pre_cem_prodmp": original,
        "pre_cem_with_zero": original or legacy_reference(controller, root, reference, controller.zero),
        "cem_fitted_prodmp": controller.free_candidate(root, reference.name, *reference.request, renderer="fit"),
    }
    if study == "primary":
        assert (proposals["cem_fitted_prodmp"] is not None) == bool(row["verification"][0]["fit_executed"])
    for candidate in proposals.values():
        if candidate is not None:
            np.testing.assert_array_equal(candidate.s, reference.s)
            np.testing.assert_array_equal(candidate.vs, reference.vs)
            for field in ("x", "v"):
                np.testing.assert_allclose(getattr(candidate.lateral, field)[[0, -1]],
                                           getattr(reference.lateral, field)[[0, -1]], atol=1e-12, rtol=0.)
    results = []
    for saved in row["verification"]:
        scene_after = tape_for(scene.latent, seed, saved["replica"], tape_namespace)
        outcomes = {}
        for arm, candidate in proposals.items():
            if candidate is None:
                # Every arm can fall back to the admitted original reference.
                outcomes[arm] = outcomes["reference_waypoint"].copy()
                continue
            metrics, _ = real_trial(controller, root, prefix, scene_after, [candidate])
            outcomes[arm] = {field: metrics[field] for field in FIELDS}
        np.testing.assert_allclose(
            [outcomes["reference_waypoint"][key] for key in FIELDS],
            [saved["outcomes"][archived_arm][key] for key in FIELDS], atol=1e-12, rtol=0.)
        if study == "primary":
            np.testing.assert_allclose(
                [outcomes["cem_fitted_prodmp"][key] for key in FIELDS],
                [saved["outcomes"]["prodmp_same_first"][key] for key in FIELDS], atol=1e-12, rtol=0.)
        results.append(outcomes)
    return {"study": study, "family": family, "seed": seed, "selected_opening": chosen,
            "request": reference.request, "feasible": {arm: c is not None for arm, c in proposals.items()},
            "replicas": results}


def summarize(rows):
    compare = ("pre_cem_with_zero", "cem_fitted_prodmp")
    groups = {
        "all_requested": rows,
        "both_prodmps_feasible": [r for r in rows if all(r["feasible"][arm] for arm in compare)],
    }
    groups.update({family: [r for r in rows if r["family"] == family]
                   for family in sorted({r["family"] for r in rows})})
    report = {"feasible_worlds": {arm: sum(r["feasible"][arm] for r in rows) for arm in ARMS},
              "requested_worlds": len(rows)}
    for name, subset in groups.items():
        if subset:
            detail = paired_statistics(subset, ARMS, FIELDS, lambda r: r["replicas"])
            contrast = paired_statistics(subset, compare, FIELDS, lambda r: r["replicas"])
            detail["cem_fitted_minus_pre_cem"] = contrast["vs_reference"]["cem_fitted_prodmp"]
            detail["completion_disagreements"] = sum(
                t[compare[0]]["corrected_complete"] != t[compare[1]]["corrected_complete"]
                for r in subset for t in r["replicas"]
            )
            detail["correct_completions"] = {arm: int(sum(t[arm]["corrected_complete"]
                                                           for r in subset for t in r["replicas"])) for arm in ARMS}
            report[name] = detail
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--limited", required=True, type=Path)
    parser.add_argument("--fit", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--jobs", default=8, type=int)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")
    init_worker(np.load(args.fit))
    jobs = []
    for study, path, namespace, arm in (("primary", args.primary, 1101, "public_first_only"),
                                         ("limited", args.limited, 1117, "public_budgeted")):
        with gzip.open(path, "rt") as stream:
            jobs.extend((study, namespace, arm, row) for row in map(json.loads, stream))
    with get_context("fork").Pool(args.jobs) as pool:
        results = [row for row in pool.imap(replay, jobs) if row is not None]
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {study: summarize([r for r in results if r["study"] == study]) for study in ("primary", "limited")}
    (args.out / "summary.json").write_text(json.dumps(summary, default=jsonable, indent=2) + "\n")
    with gzip.open(args.out / "cases.jsonl.gz", "wt", compresslevel=3) as stream:
        for row in results:
            stream.write(json.dumps(row, default=jsonable, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
