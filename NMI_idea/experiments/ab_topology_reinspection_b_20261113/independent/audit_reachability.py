"""Evaluator-only upper envelope for fresh target acquisitions on a fixed tape."""
import argparse
import json
from itertools import combinations_with_replacement
from pathlib import Path
import sys

import numpy as np

from audit_trials import fine_to_coarse, observe, records, uniform_tape


def advance(next_layer, key, resources):
    frontier = next_layer.setdefault(key, [])
    if any(np.all(other <= resources + 1e-10) for other in frontier):
        return
    frontier[:] = [other for other in frontier if not np.all(resources <= other + 1e-10)]
    frontier.append(resources)


def shortest_witness(edges, sites, at, start_clock, horizon, near_site, caps, visible):
    """Exact terminal reachability using joint Pareto costs at each physical state."""
    layer = {(at, False): [np.zeros(3)]}
    for dt in range(1, horizon + 1):
        next_layer = {}
        clock = start_clock + dt
        for (node, previous_is_usable_near), used_values in layer.items():
            goals, costs = edges[node]
            for used in used_values:
                reachable = np.flatnonzero(np.all(used + costs <= caps + 1e-8, axis=1))
                for k in reachable:
                    goal = int(goals[k])
                    site = sites[goal]
                    usable = visible[clock, goal]
                    if previous_is_usable_near and site == near_site - 1 and usable:
                        return dt
                    advance(next_layer, (goal, bool(site == near_site and usable)), used + costs[k])
        layer = next_layer
        if not layer:
            break
    return None


def inspect(world, geometry, records_path, cohort, count, grid, prefix, suffix, caps):
    world_seed, namespace, split = cohort
    case_batch = world.cases(geometry, world_seed, count, split)
    mapping = fine_to_coarse(grid)
    numbers = {packet: i for i, packet in enumerate(combinations_with_replacement(range(1 + 3 * grid), 3))}
    edges = [(legal := np.flatnonzero(np.isfinite(geometry.cost[at, :, 0])), geometry.cost[at, legal])
             for at in range(len(geometry.sites))]
    output = []
    for _, record in records(records_path):
        index = record["index"]
        label = int(case_batch.labels[index])
        if label <= 0:
            continue
        near = 4 + (label - 1) // 2
        at = int(record["prefix"][-1][0])
        tape = uniform_tape(namespace, index, (len(geometry.clock_s), 5, 7))
        visibility = np.zeros((len(geometry.clock_s), len(geometry.sites)), dtype=bool)
        for clock in range(prefix + 1, prefix + suffix + 1):
            for node, site in enumerate(geometry.sites):
                if site not in (near, near - 1):
                    continue
                _, kinds = observe(case_batch, index, clock, node, site, tape, mapping, numbers)
                visibility[clock, node] = 0 in kinds
        mechanical = shortest_witness(edges, geometry.sites, at, prefix, suffix, near, caps, np.ones_like(visibility))
        clairvoyant = shortest_witness(edges, geometry.sites, at, prefix, suffix, near, caps, visibility)
        assert mechanical is not None or clairvoyant is None
        output.append({
            "index": index,
            "near_site": near,
            "prefix_exit": at,
            "minimum_suffix_steps_mechanical": mechanical,
            "minimum_suffix_steps_clairvoyant_tape": clairvoyant,
            "actual_baseline_success": bool(record["success"]),
        })
    assert len(output) == int(np.count_nonzero(case_batch.labels))
    return {
        "interpretation": "Fixed-tape, truth-assisted feasibility ceiling. Not an online policy or a population optimum.",
        "positive_cases": len(output),
        "mechanically_feasible": sum(row["minimum_suffix_steps_mechanical"] is not None for row in output),
        "clairvoyant_tape_feasible": sum(row["minimum_suffix_steps_clairvoyant_tape"] is not None for row in output),
        "actual_baseline_success": sum(row["actual_baseline_success"] for row in output),
        "rows": output,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage", "repo", "settings", "records", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.stage.resolve()))
    import world

    geometry = world.load(args.repo)
    settings = json.loads(args.settings.read_text())
    report = inspect(world, geometry, args.records, settings["cohort"], settings["cases"],
                     settings["grid"], settings["prefix"], settings["suffix"], np.asarray(settings["suffix_caps"]))
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
