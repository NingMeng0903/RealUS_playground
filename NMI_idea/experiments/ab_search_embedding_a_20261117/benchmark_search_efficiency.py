"""Measure first verified recovery under an equal closed-loop simulation budget."""

import argparse
from collections import Counter
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
from time import process_time

import numpy as np

from adaptive_teacher import (SEARCH_ANCHORS, SEARCH_LIMITS, SEARCH_SCALE, fit_nominal_primitive,
                              refit_search_distribution, utility)
from diagnose_b_selection import run_parallel, save_archive
from run_adaptive_recovery import paired_statistics, tape_for
from run_learned_recovery import CONTROLLERS, FAMILIES, configure, fingerprint, read_rows, write_json
from verify_recovery_components import bands
from verify_startup_ood import OUTCOMES, execute_trial, public_choices, render_action, scout, screen, startup_controller


SPLITS = {"discovery": 42_310_000, "holdout": 43_310_000}
GROUPS = ("zero", "one")
METHODS = ("random", "cem")
SCREEN_PER_FAMILY = 320
QUOTA_PER_CELL = 16
ROUNDS = 4
POPULATION = 4
SCORING_REPEATS = 3
VERIFY_FINALISTS = 2
VERIFY_REPEATS = 6
AUDIT_REPEATS = 12
HOLDOUT_REPEATS = 8
ROLLOUTS_PER_ROUND = len(SEARCH_ANCHORS) * POPULATION * SCORING_REPEATS + VERIFY_FINALISTS * VERIFY_REPEATS
BUDGET = ROLLOUTS_PER_ROUND * ROUNDS
NAMESPACES = {"score": 15301, "confirm": 15311, "audit": 15321, "holdout": 15331, "mpc": 15341}
MAX_PROPOSALS = 4096
REQUESTS_PER_SOURCE = 6
TRACE_FIELDS = ("corrected_complete", "final_wrong", "new_identity_mistake", "first_interval_recaptured",
                "tcp_mm", "turn_degs", "seconds")


def screen_cases(out, jobs):
    report = {}
    for split, namespace in SPLITS.items():
        jobs_to_screen = [(family, namespace + 10_000 * i + j, "holdout")
                          for i, family in enumerate(FAMILIES) for j in range(SCREEN_PER_FAMILY)]
        rows = run_parallel(screen, jobs_to_screen, jobs, f"{split} screening")
        selected = []
        for family, group in product(FAMILIES, GROUPS):
            observed = [] if group == "zero" else [1]
            eligible = [r for r in rows if r["family"] == family and r["observed_vessels"] == observed]
            if len(eligible) < QUOTA_PER_CELL:
                raise ValueError(f"{split} has {len(eligible)} eligible {family}/{group} starts")
            selected.extend(eligible[:QUOTA_PER_CELL])
        save_archive(out / f"{split}_screening.jsonl.gz", rows)
        save_archive(out / f"{split}_starts.jsonl.gz", selected)
        report[split] = {"total_screened": len(rows), "selected": len(selected),
                         "blank_scouts": sum(r["blank_packets"] == 11 for r in selected)}
    write_json(out / "screening_summary.json", report)


def replay_bank(controller, root, prefix, scene, seed, action, purpose, repeats, *, offset=0):
    outcomes = []
    for i in range(offset, offset + repeats):
        tape = tape_for(scene.latent, seed, i, NAMESPACES[purpose]).tape
        metrics = execute_trial(controller, root, prefix, replace(scene, tape=tape), action)
        outcomes.append({key: metrics[key] for key in TRACE_FIELDS})
    return outcomes


def verified(outcomes, minimum):
    return (sum(t["corrected_complete"] for t in outcomes) >= minimum
            and not any(t["final_wrong"] for t in outcomes))


def draw_population(controller, root, mean, scale, rng, scored):
    """Charge simulations only for mechanically valid, distinct paths."""
    population = []
    for attempts in range(1, MAX_PROPOSALS + 1):
        request = np.clip(rng.normal(mean, scale), *SEARCH_LIMITS)
        candidate = controller.free_candidate(root, "offline_probe", *request)
        if candidate is None:
            continue
        if any(np.max(abs(old.s - candidate.s)) < .05 and np.max(abs(old.lateral.x - candidate.lateral.x)) < .05
               for old in scored):
            continue
        population.append(candidate)
        scored.append(candidate)
        if len(population) == POPULATION:
            return population, attempts
    raise RuntimeError("could not match the budget of distinct feasible trajectories")


def search(controller, root, prefix, scene, seed, method):
    rng = np.random.default_rng([seed, 15601])
    means, scales = SEARCH_ANCHORS.copy(), np.tile(SEARCH_SCALE, (len(SEARCH_ANCHORS), 1))
    attempts, used = 0, 0
    first, scored, rounds = None, [], []
    for index in range(ROUNDS):
        round_entries = []
        for anchor, (mean, scale) in enumerate(zip(means, scales, strict=True)):
            population, proposed = draw_population(controller, root, mean, scale, rng, scored)
            attempts += proposed
            scored_population = []
            for action in population:
                trials = replay_bank(controller, root, prefix, scene, seed, action, "score", SCORING_REPEATS,
                                     offset=index * SCORING_REPEATS)
                used += len(trials)
                score = float(np.mean([utility(t) for t in trials]))
                scored_population.append((score, action, trials, anchor))
            scored_population.sort(key=lambda row: row[0], reverse=True)
            round_entries.extend(scored_population)
            if method == "cem":
                elite = np.array([action.request for _, action, _, _ in scored_population[:2]])
                means[anchor], scales[anchor] = refit_search_distribution(mean, scale, elite)

        round_entries.sort(key=lambda row: row[0], reverse=True)
        finalists = []
        for rank, (_, action, _, _) in enumerate(round_entries[:VERIFY_FINALISTS]):
            trials = replay_bank(controller, root, prefix, scene, seed, action, "confirm", VERIFY_REPEATS,
                                 offset=index * VERIFY_REPEATS)
            used += len(trials)
            passed = verified(trials, 4)
            finalists.append({"rank": rank, "after_rollouts": used, "trials": trials, "qualified": passed})
            if first is None and passed:
                first = (action, used)
        if used != (index + 1) * ROLLOUTS_PER_ROUND:
            raise AssertionError("unmatched simulation budget")
        rounds.append({"scored": [{"anchor": anchor, "request": action.request, "utility": score, "trials": trials}
                                   for score, action, trials, anchor in round_entries],
                       "finalists": finalists, "proposals_to_date": attempts, "rollouts_to_date": used})

    record = {"rounds": rounds, "proposals": attempts, "rollouts": used, "first": None}
    if first is not None:
        action, at = first
        trials = replay_bank(controller, root, prefix, scene, seed, action, "audit", AUDIT_REPEATS)
        record["first"] = {"request": action.request, "at_rollouts": at,
                           "nominal_x_vx": np.column_stack((action.lateral.x, action.lateral.v)),
                           "audit": trials, "audit_pass": verified(trials, 8)}
    return record


def discover(row):
    controller = CONTROLLERS[row["family"]]
    seed = row["seed"]
    scene = controller.world.startup_draw(seed, "holdout")
    root, prefix = scout(controller, scene)
    baseline = controller.choose_mpc(root, expanded=True, open_decision=True)
    mpc = replay_bank(controller, root, prefix, scene, seed, baseline, "mpc", 8)
    results = {}
    for method in METHODS:
        started = process_time()
        results[method] = search(controller, root, prefix, scene, seed, method)
        results[method]["cpu_seconds_with_audit"] = process_time() - started
    if results["random"]["rounds"][0] != results["cem"]["rounds"][0]:
        raise AssertionError("the common first round differs between search methods")
    return row | {"mpc_screen": mpc, "search": results}


def discovery_metrics(record):
    first = record["first"]
    at = first["at_rollouts"] if first else BUDGET + 1
    return {f"verified_by_{budget}": int(at <= budget)
            for budget in range(ROLLOUTS_PER_ROUND, BUDGET + 1, ROLLOUTS_PER_ROUND)} | {
        "audit_verified": int(first is not None and first["audit_pass"]),
        "rollouts_to_first_capped": min(at, BUDGET), "proposals": record["proposals"],
        "cpu_seconds_with_audit": record["cpu_seconds_with_audit"],
    }


def summarize_discovery(rows):
    def extract(row):
        return [{method: discovery_metrics(row["search"][method]) for method in METHODS}]

    difficult = [r for r in rows if sum(t["corrected_complete"] for t in r["mpc_screen"]) <= 6]
    cohorts = {"all": rows, "mpc_difficult": difficult}
    cohorts.update({f"{family}_{group}": [r for r in rows if r["family"] == family and r["group"] == group]
                    for family, group in product(FAMILIES, GROUPS)})
    report = {}
    for label, cohort in cohorts.items():
        if not cohort:
            continue
        values = [extract(r)[0] for r in cohort]
        fields = tuple(values[0]["random"])
        info = paired_statistics(cohort, METHODS, fields, extract)
        verified_fields = [name for name in fields if name.startswith("verified_by_")] + ["audit_verified"]
        info["counts"] = {method: {field: sum(v[method][field] for v in values) for field in verified_fields}
                          for method in METHODS}
        info["paired_audit"] = {f"{c}/{r}": n for (c, r), n in sorted(Counter(
            (v["cem"]["audit_verified"], v["random"]["audit_verified"]) for v in values).items())}
        report[label] = info
    return report


def audit_pass(row, method):
    first = row["search"][method]["first"]
    return first is not None and first["audit_pass"]


def fit_and_freeze(rows, out):
    paired = [row for row in rows if all(audit_pass(row, method) for method in METHODS)]
    primitive = CONTROLLERS[FAMILIES[0]].primitive
    fits, libraries = {}, {}
    for method in METHODS:
        fits[method] = fit_nominal_primitive(primitive, [r["search"][method]["first"] for r in paired])
        if fits[method] is None:
            raise ValueError("too few independent successful training requests for matched ProDMP fitting")
        np.save(out / f"{method}_fit.npy", fits[method])
        winners = [r["search"][method]["first"] for r in rows if audit_pass(r, method)]
        winners.sort(key=lambda d: np.mean([utility(t) for t in d["audit"]]), reverse=True)
        libraries[method] = list(dict.fromkeys(tuple(d["request"]) for d in winners))
    size = min(REQUESTS_PER_SOURCE, *(len(menu) for menu in libraries.values()))
    write_json(out / "bundle.json", {
        "requests": {method: requests[:size] for method, requests in libraries.items()},
        "fit_seeds": [r["seed"] for r in paired],
        "fit_sha256": {method: fingerprint(out / f"{method}_fit.npy") for method in METHODS},
        "max_coefficient_difference": float(np.max(abs(fits["cem"] - fits["random"]))),
        "discovery_archive_sha256": fingerprint(out / "discovery.jsonl.gz"),
    })


def run_discovery(out, jobs):
    rows = run_parallel(discover, read_rows(out / "discovery_starts.jsonl.gz"), jobs, "paired search")
    save_archive(out / "discovery.jsonl.gz", rows)
    write_json(out / "discovery_summary.json", {
        "rollout_budget_per_scene_and_method": BUDGET, "rollouts_per_round": ROLLOUTS_PER_ROUND,
        "total_search_rollouts": {method: sum(r["search"][method]["rollouts"] for r in rows) for method in METHODS},
        "cohorts": summarize_discovery(rows),
    })
    fit_and_freeze(rows, out)


def audit_remaining_case(row):
    controller = CONTROLLERS[row["family"]]
    seed = row["seed"]
    scene = controller.world.startup_draw(seed, "holdout")
    root, prefix = scout(controller, scene)
    result = {}
    for method in METHODS:
        search_record = row["search"][method]
        first = search_record["first"]
        finalists = []
        fresh = 0
        for batch in search_record["rounds"]:
            for finalist in batch["finalists"]:
                request = batch["scored"][finalist["rank"]]["request"]
                at = finalist["after_rollouts"]
                if at == first["at_rollouts"]:
                    trials = first["audit"]
                else:
                    candidate = controller.free_candidate(root, "offline_probe", *request)
                    if candidate is None:
                        raise ValueError("archived finalist is no longer mechanically feasible")
                    trials = replay_bank(controller, root, prefix, scene, seed, candidate, "audit", AUDIT_REPEATS)
                    fresh += len(trials)
                finalists.append({
                    "request": request, "at_search_rollouts": at, "qualification_pass": finalist["qualified"],
                    "audit": trials, "audit_pass": verified(trials, 8),
                })
        first_robust = next((f["at_search_rollouts"] for f in finalists if f["qualification_pass"]
                             and f["audit_pass"]), None)
        result[method] = {"first_replicated_at_search_rollouts": first_robust, "fresh_audit_rollouts": fresh,
                          "finalists": finalists}
    if result["random"]["fresh_audit_rollouts"] != result["cem"]["fresh_audit_rollouts"]:
        raise AssertionError("unmatched follow-up audit cost")
    return {"seed": seed, "family": row["family"], "group": row["group"], "search_audit": result}


def run_remaining_audit(out, jobs):
    path = out / "discovery.jsonl.gz"
    eligible = [r for r in read_rows(path) if all(
        r["search"][m]["first"] is not None and not audit_pass(r, m) for m in METHODS)]
    rows = run_parallel(audit_remaining_case, eligible, jobs, "remaining finalist audit") if eligible else []
    save_archive(out / "remaining_finalists.jsonl.gz", rows)
    write_json(out / "remaining_finalists_summary.json", {
        "discovery_archive_sha256": fingerprint(path), "cases": len(rows),
        "additional_audit_calls": {m: sum(r["search_audit"][m]["fresh_audit_rollouts"] for r in rows)
                                   for m in METHODS},
        "rescued": {m: sum(r["search_audit"][m]["first_replicated_at_search_rollouts"] is not None for r in rows)
                    for m in METHODS},
    })


def holdout_case(job):
    row, bundle, fits = job
    controller = CONTROLLERS[row["family"]]
    seed = row["seed"]
    scene = controller.world.startup_draw(seed, "holdout")
    root, prefix = scout(controller, scene)
    choices = public_choices(controller, root, bundle)
    plans = {name: (choices[name], "original") for name in ("mpc", "stock")}
    max_path_difference = {}
    for menu in METHODS:
        original = choices[menu]
        plans[f"{menu}_analytic"] = (original, "original")
        for fitter in METHODS:
            label = f"{menu}_{fitter}fit"
            plans[label] = render_action(controller, root, original, f"{fitter}fit", fits)
            plan, used = plans[label]
            if used == f"{fitter}fit":
                max_path_difference[label] = float(np.max(abs(plan.lateral.x - original.lateral.x)))
    verification = []
    for i in range(HOLDOUT_REPEATS):
        tape = tape_for(scene.latent, seed, i, NAMESPACES["holdout"]).tape
        future = replace(scene, tape=tape)
        cache, outcomes = {}, {}
        for label, (action, _) in plans.items():
            key = id(action)
            if key not in cache:
                cache[key] = execute_trial(controller, root, prefix, future, action)
            outcomes[label] = cache[key]
        verification.append(outcomes)
    return row | {"chosen": {k: {"name": v.name, "request": getattr(v, "request", None)}
                             for k, v in choices.items()},
                  "renderer": {name: kind for name, (_, kind) in plans.items()},
                  "fit_max_position_error_mm": max_path_difference,
                  "verification": verification}


def run_holdout(out, jobs):
    bundle = json.loads((out / "bundle.json").read_text())
    fits = {}
    for method in METHODS:
        path = out / f"{method}_fit.npy"
        if fingerprint(path) != bundle["fit_sha256"][method]:
            raise ValueError(f"{method} ProDMP fit differs from the frozen bundle")
        fits[f"{method}fit"] = np.load(path, allow_pickle=False)
    starts = read_rows(out / "holdout_starts.jsonl.gz")
    if {r["seed"] for r in starts} & set(bundle["fit_seeds"]):
        raise ValueError("holdout scene reused for ProDMP fitting")
    rows = run_parallel(holdout_case, [(r, bundle, fits) for r in starts], jobs, "ProDMP holdout")
    save_archive(out / "holdout.jsonl.gz", rows)
    contrasts = {
        "complete_stacks": ("mpc", "stock", "random_randomfit", "cem_cemfit"),
        "candidate_source_fixed_renderer": ("random_analytic", "cem_analytic"),
        "random_menu_same_request": ("random_analytic", "random_randomfit", "random_cemfit"),
        "cem_menu_same_request": ("cem_analytic", "cem_randomfit", "cem_cemfit"),
    }
    report = {name: bands(rows, arms, OUTCOMES) for name, arms in contrasts.items()}
    report["rendering"] = {
        arm: dict(Counter(r["renderer"][arm] for r in rows)) for arm in rows[0]["renderer"]
    }
    report["maximum_direct_fit_position_error_mm"] = max(
        (e for r in rows for e in r["fit_max_position_error_mm"].values()), default=None)
    report["bundle_sha256"] = fingerprint(out / "bundle.json")
    report["holdout_archive_sha256"] = fingerprint(out / "holdout.jsonl.gz")
    write_json(out / "holdout_summary.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("screen", "discover", "holdout", "audit_remaining"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    configure(Path("results/visible_atlas_v1"), controller_type=startup_controller)
    phases = {"screen": screen_cases, "discover": run_discovery, "holdout": run_holdout,
              "audit_remaining": run_remaining_audit}
    phases[args.phase](args.out, args.jobs)


if __name__ == "__main__":
    main()
