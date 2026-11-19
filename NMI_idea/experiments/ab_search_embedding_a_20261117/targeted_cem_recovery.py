"""Search a missing-interval segment offline, then verify it on fresh paired tapes."""

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from adaptive_controller import EDGES
from budgeted_recovery import CAPS
from diagnose_b_selection import run_parallel, save_archive
from learned_recovery import OBJECTIVE, TARGETS, outcome_vector
from run_adaptive_recovery import real_trial, shared_prefix, tape_for
from run_learned_recovery import CONTROLLERS, config, configure, fingerprint, read_rows, write_json
from run_qualified_recovery import NO_IMAGE, interval_failures, summarize


ARMS = ("mpc", "teacher", "spatial_cem", "timed_cem", "timed_cem_without_timing")
BOUNDS = np.array([[-6., -8.5, -3.7, -2.4], [7.8, 8.5, 3.7, 2.4]])
SCALE = np.array([2.3, 2.7, 1.3, 1.2])
SEARCH_TAPE = 5843
VERIFY_TAPE = 5849


def passes_caps(delta):
    return all(delta[TARGETS.index(field)] <= cap + 1e-8 for field, cap in CAPS.items())


def search(controller, root, prefix, worlds, incumbent, target, seed, population, iterations, dimension):
    baseline = np.mean([outcome_vector(real_trial(controller, root, prefix, scene, [incumbent])[0])
                        for scene in worlds], axis=0)
    station = float(np.mean(EDGES[target:target + 2]))
    nominal = controller.world.target_grid(station, root.angle)[worlds[0].latent, 0]
    location = np.clip([station - root.s, nominal - root.x, 0., 0.], *BOUNDS)
    mean, scale = location[:dimension].copy(), SCALE[:dimension].copy()
    rng = np.random.default_rng([seed, 2771])
    archive = []
    best, best_score = None, 0.
    for _ in range(iterations):
        draws = np.clip(rng.normal(mean, scale, (population, dimension)), *BOUNDS[:, :dimension])
        draws[0] = mean
        ranked = []
        for draw in draws:
            action = controller.free_candidate(root, "cem_targeted", *draw, renderer="fit")
            if action is None:
                archive.append({"request": draw, "feasible_motion": False})
                continue
            outcomes = [outcome_vector(real_trial(controller, root, prefix, scene, [action])[0])
                        for scene in worlds]
            delta = np.mean(outcomes, axis=0) - baseline
            score = float(delta @ OBJECTIVE)
            allowed = passes_caps(delta)
            archive.append({"request": draw, "feasible_motion": True, "passes_caps": allowed,
                            "utility_gain": score, "paired_outcomes": outcomes, "delta": delta})
            ranked.append((allowed, score, draw))
            if allowed and score > best_score + 1e-9:
                best, best_score = action, score
        if ranked:
            elite = np.stack([row[2] for row in
                              sorted(ranked, key=lambda row: row[:2], reverse=True)[:max(2, population // 4)]])
            mean = .3 * mean + .7 * elite.mean(axis=0)
            scale = .3 * scale + .7 * (elite.std(axis=0) + .25)
    return best, {"baseline": baseline, "best_score": best_score, "evaluated": archive}


def verify_case(job):
    screened, target, args = job
    family, seed = screened["family"], screened["seed"]
    controller = CONTROLLERS[family]
    scene = controller.world.draw(seed)
    root, prefix = shared_prefix(controller, scene)
    menu = controller.options_shared(root, renderer="fit")
    mpc, teacher = (menu[screened["selected"][arm]] for arm in ARMS[:2])
    search_worlds = [tape_for(scene.latent, seed, k, SEARCH_TAPE) for k in range(args.samples)]
    spatial, spatial_log = search(controller, root, prefix, search_worlds, mpc, target, seed,
                                  args.population, args.iterations, 3)
    timed, timed_log = search(controller, root, prefix, search_worlds, mpc, target, seed,
                              args.population, args.iterations, 4)
    untimed = (controller.free_candidate(root, "cem_targeted", *timed.request[:3], renderer="fit")
               if timed is not None else None)
    candidates = dict(zip(ARMS, (mpc, teacher, spatial or mpc, timed or mpc, untimed or mpc), strict=True))
    trials = []
    for k in range(args.replicas):
        future = tape_for(scene.latent, seed, k, VERIFY_TAPE)
        outcomes = {}
        for arm, action in candidates.items():
            metrics, record = real_trial(controller, root, prefix, future, [action])
            outcomes[arm] = metrics | {"arrived": record.motion[-1][1] >= EDGES[-1] - .28,
                                       "intervals": interval_failures(controller, scene, record, metrics)}
        trials.append({"replica": k, "outcomes": outcomes})

    proposals = {}
    means = {arm: np.mean([outcome_vector(t["outcomes"][arm]) for t in trials], axis=0) for arm in ARMS}
    for label, action in (("spatial_cem", spatial), ("timed_cem", timed)):
        if action is None or (label == "timed_cem" and untimed is None):
            continue
        delta = means[label] - means["mpc"]
        needed = sum(t["outcomes"][label]["intervals"][target] == "qualified"
                     and t["outcomes"]["mpc"]["intervals"][target] in NO_IMAGE for t in trials)
        if (delta[0] >= 2 / len(trials) and means[label][0] > means["teacher"][0]
                and passes_caps(delta) and needed >= 2
                and (label == "spatial_cem" or means[label][0] > means["timed_cem_without_timing"][0])):
            proposals[label] = {"request": action.request, "new_target_captures": needed, "delta": delta}
    return {"family": family, "seed": seed, "target_bin_zero_based": target,
            "selected_requests": {label: getattr(action, "request", None) for label, action in candidates.items()},
            "exact_zero_timing_available": untimed is not None, "timed_search_selected": timed is not None,
            "search": {"spatial": spatial_log, "timed": timed_log},
            "proposals": proposals, "verification": trials}


def replay_higher_cost_candidate(row):
    """Test whether a predicted recovery survives when only the hard motion bounds apply."""
    controller = CONTROLLERS[row["family"]]
    seed = row["seed"]
    scene = controller.world.draw(seed)
    root, prefix = shared_prefix(controller, scene)
    positive = [draw for search_log in row["search"].values() for draw in search_log["evaluated"]
                if draw["feasible_motion"] and draw["delta"][0] > 0.]
    best = max(positive, key=lambda draw: draw["utility_gain"], default=None)
    if best is None:
        return {"seed": seed, "verification": []}
    action = controller.free_candidate(root, "cem_targeted", *best["request"], renderer="fit")
    if action is None:
        raise ValueError("archived candidate no longer satisfies the motion checks")
    verification = []
    for saved in row["verification"]:
        k = saved["replica"]
        future = tape_for(scene.latent, seed, k, VERIFY_TAPE)
        metrics, record = real_trial(controller, root, prefix, future, [action])
        result = metrics | {"arrived": record.motion[-1][1] >= EDGES[-1] - .28,
                            "intervals": interval_failures(controller, scene, record, metrics)}
        verification.append({"replica": k, "outcomes": {"mpc": saved["outcomes"]["mpc"], "higher_cost_cem": result}})
    return {"family": row["family"], "seed": seed, "target_bin_zero_based": row["target_bin_zero_based"],
            "selection_from_search_tapes": best, "verification": verification}


def audit_costs(archive, out, jobs):
    rows = read_rows(archive)
    results = run_parallel(replay_higher_cost_candidate, rows, jobs, "higher-cost replay")
    results = [row for row in results if row["verification"]]
    save_archive(out / "higher_cost_cases.jsonl.gz", results)
    write_json(out / "higher_cost_summary.json", summarize(results, ("mpc", "higher_cost_cem")) | {
        "targeted_archive_sha256": fingerprint(archive), "screened_cases": len(rows),
        "cases_with_positive_search_draws": len(results), "verification_tape": VERIFY_TAPE,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--higher-cost-replay", action="store_true",
                        help="replay candidates from a targeted CEM archive without the relative cost caps")
    parser.add_argument("--cases", type=int, default=24)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=20)
    args = parser.parse_args()
    if min(args.samples, args.replicas, args.population, args.iterations, args.jobs, args.cases) < 1:
        parser.error("positive search and verification counts required")
    if args.replicas < 2:
        parser.error("verification requires repeated observations")
    args.out.mkdir(parents=True, exist_ok=True)
    configure(args.source)
    audit = json.loads((args.audit.parent / "summary.json").read_text())
    if (fingerprint(args.source / "library.json") != audit["library_sha256"]
            or fingerprint(args.audit) != audit["archive_sha256"]):
        parser.error("screening archive and shared library must match the audited run")
    if args.higher_cost_replay:
        audit_costs(args.audit, args.out, args.jobs)
        return
    candidates = []
    for row in read_rows(args.audit):
        if row["family"] != "parallel" or row["selected"]["mpc"] is None:
            continue
        misses = Counter(j for trial in row["verification"]
                         for j, reason in enumerate(trial["outcomes"]["teacher"]["intervals"])
                         if reason in NO_IMAGE)
        repeated = [(count, j) for j, count in misses.items() if count >= 2]
        if repeated:
            count, target = min(repeated, key=lambda pair: (-pair[0], pair[1]))
            candidates.append((count, row, target))
    selected = sorted(candidates, key=lambda row: (-row[0], row[1]["seed"]))[:args.cases]
    jobs = [(row, target, args) for _, row, target in selected]
    started = perf_counter()
    rows = run_parallel(verify_case, jobs, args.jobs, "targeted CEM") if jobs else []
    archive = args.out / "cases.jsonl.gz"
    save_archive(archive, rows)
    nominees = [{"seed": row["seed"], "arm": arm, **result}
                for row in rows for arm, result in row["proposals"].items()]
    ranked = sorted(nominees, key=lambda item: (-item["delta"][0], -item["new_target_captures"], item["seed"]))
    unique = list(dict.fromkeys(tuple(item["request"]) for item in ranked))
    write_json(args.out / "summary.json", summarize(rows, ARMS) | {
        "config": config(args), "eligible_cases": len(candidates), "screened_cases": len(rows),
        "timed_motion_feasible_in_both_forms": sum(row["exact_zero_timing_available"] for row in rows),
        "nominees": nominees, "shared_proposals": unique[:4], "library_sha256": audit["library_sha256"],
        "screening_sha256": audit["archive_sha256"], "archive_sha256": fingerprint(archive),
        "elapsed_s": perf_counter() - started,
    })


if __name__ == "__main__":
    main()
