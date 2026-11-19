"""Compare CEM and equal-budget random proposals on genuine acquisition misses."""

import argparse
from collections import Counter
import json
from pathlib import Path
from shutil import copyfile

import numpy as np

from adaptive_controller import BINS, station_bin
from adaptive_teacher import cem_discover, evaluate, expected, fixed_worlds, posterior_worlds, utility
from diagnose_b_selection import run_parallel, save_archive
from learned_recovery import OBJECTIVE, choose_from_deltas, outcome_vector
from run_adaptive_recovery import real_trial, tape_for
from run_learned_recovery import FAMILIES, configure, fingerprint, read_rows, write_json
from verify_recovery_components import (CAPS_WITHOUT_RELATIVE_MOTION, FIELDS, TARGETS, bands, case_start,
                                        rollout_scores, stock_controller, within_caps)


SHADOW_SCALE = 3.0
SCREEN_BASE = {"development": 31_410_000, "holdout": 33_410_000}
SCREEN_COUNT = {"development": 192, "holdout": 256}
HARD_COUNT = {"development": 24, "holdout": 32}
SCREEN_TAPE = 9211
DEV_TAPE = 9212
TEST_TAPE = 9213
SCREEN_REPEATS = 4
VERIFY_REPEATS = 8
SEARCH_ROLLOUTS = 3
PUBLIC_ROLLOUTS = 12
ARMS = ("mpc", "fixed_revisit", "public_stock", "public_random", "public_cem")
DEV_ARMS = ("mpc", "fixed_revisit", "privileged_stock", "privileged_random", "privileged_cem")
SOURCES = ("random", "cem")
STATS_FIELDS = (*FIELDS, "captured_all_baseline_gaps", "remaining_baseline_gaps")


def stock_options(family, seed):
    controller, scene, root, prefix = case_start(family, seed)
    menu = controller.options_shared(root)
    incumbent = controller.mpc(root, menu, False, active=True, informative=True).name
    return controller, scene, root, prefix, menu, incumbent


def acquired_intervals(record):
    return {station_bin(p["s"]) for p in record.packets if 0 in p["roles_truth"]} - {-1}


def screen(job):
    family, seed = job
    c, scene, root, prefix, menu, incumbent = stock_options(family, seed)
    attempts = []
    for index in range(SCREEN_REPEATS):
        future = tape_for(scene.latent, seed, index, SCREEN_TAPE)
        result, record = real_trial(c, root, prefix, future, [menu[incumbent]])
        attempts.append({"missing_intervals": sorted(set(range(BINS)) - acquired_intervals(record)),
                         "corrected_complete": result["corrected_complete"]})
    return {"family": family, "seed": seed, "screening": attempts}


def run_screen(args):
    result = {}
    for split, start in SCREEN_BASE.items():
        jobs = [(family, start + 100_000 * i + j) for i, family in enumerate(FAMILIES)
                for j in range(SCREEN_COUNT[split])]
        rows = run_parallel(screen, jobs, args.jobs, f"{split} acquisition screening")
        hard = [r for r in rows if any(t["missing_intervals"] for t in r["screening"])]
        selected = []
        for family in FAMILIES:
            selected.extend([row for row in hard if row["family"] == family][:HARD_COUNT[split]])
        save_archive(args.out / f"{split}_screening.jsonl.gz", rows)
        save_archive(args.out / f"{split}_hard_cases.jsonl.gz", selected)
        result[split] = {
            "screened": dict(Counter(r["family"] for r in rows)),
            "has_a_genuine_miss": dict(Counter(r["family"] for r in hard)),
            "selected_in_seed_order": dict(Counter(r["family"] for r in selected)),
            "missed_intervals": dict(Counter(j for r in hard for t in r["screening"] for j in t["missing_intervals"])),
        }
    write_json(args.out / "screening_summary.json", result | {"shadow_scale": SHADOW_SCALE})


def execute_paths(c, scene, root, prefix, seed, paths, tape_namespace):
    output = []
    for index in range(VERIFY_REPEATS):
        future = tape_for(scene.latent, seed, index, tape_namespace)
        cached = {}
        measured = {}
        captures = {}
        for name, action in paths.items():
            key = id(action)
            if key not in cached:
                run, record = real_trial(c, root, prefix, future, [action])
                if run["max_handoff_error"] >= 1e-7:
                    raise ValueError("motion handoff was discontinuous")
                cached[key] = ({k: run[k] for k in FIELDS}, acquired_intervals(record))
            measured[name], captures[name] = cached[key]
        missing = set(range(BINS)) - captures["mpc"]
        result = {}
        for name, metrics in measured.items():
            remaining = missing - captures[name]
            result[name] = metrics | {"captured_all_baseline_gaps": bool(missing) and not remaining,
                                      "remaining_baseline_gaps": len(remaining),
                                      "acquired_intervals": sorted(captures[name])}
        output.append({"mpc_missing_intervals": sorted(missing), "arms": result})
    return output


def discover(row):
    family, seed = row["family"], row["seed"]
    c, scene, root, prefix, menu, incumbent = stock_options(family, seed)
    scenarios = fixed_worlds(scene.latent, SEARCH_ROLLOUTS, seed, 471)
    scores = {name: expected(evaluate(c, root, action, scenarios, open_blocks=1)[0])
              for name, action in menu.items()}
    baseline = scores[incumbent]
    stock = max((name for name, value in scores.items() if within_caps(value, baseline, CAPS_WITHOUT_RELATIVE_MOTION)),
                key=lambda name: (utility(scores[name]), name == incumbent))
    paths = {"mpc": menu[incumbent], "fixed_revisit": menu.get("revisit", menu[incumbent]),
             "privileged_stock": menu[stock]}
    records = {}
    for source in SOURCES:
        _, _, ranked, draws = cem_discover(c, root, scene.latent, seed, population=6, iterations=2,
                                           samples=SEARCH_ROLLOUTS, refit=source == "cem")
        allowed = [(a, report) for a, report in ranked
                   if within_caps(report["expected"], baseline, CAPS_WITHOUT_RELATIVE_MOTION)]
        proposal = allowed[0][0] if allowed else None
        paths[f"privileged_{source}"] = (proposal if proposal is not None
                                         and allowed[0][1]["utility"] > utility(scores[stock]) else menu[stock])
        paths[f"nomination_{source}"] = proposal or menu[stock]
        records[source] = {"proposals": len(draws), "admissible": len(ranked), "admissible_with_caps": len(allowed),
                           "request": None if proposal is None else proposal.request,
                           "used_proposal": paths[f"privileged_{source}"] is proposal}
    outcomes = execute_paths(c, scene, root, prefix, seed, paths, DEV_TAPE)
    return {"family": family, "seed": seed, "discovery": records, "outcomes": outcomes,
            "selected": {name: action.name for name, action in paths.items()}}


def ordered_nominations(rows, source):
    candidates = []
    for r in rows:
        request = r["discovery"][source]["request"]
        if request is None:
            continue
        pair = np.asarray([
            np.asarray(outcome_vector(t["arms"][f"nomination_{source}"], TARGETS))
            - outcome_vector(t["arms"]["privileged_stock"], TARGETS) for t in r["outcomes"]
        ])
        delta = pair.mean(axis=0)
        recovered = sum(t["arms"][f"nomination_{source}"]["captured_all_baseline_gaps"]
                        for t in r["outcomes"])
        candidates.append((float(delta[0]), recovered, float(delta @ OBJECTIVE), tuple(request)))
    return list(dict.fromkeys(r[-1] for r in sorted(candidates, reverse=True)))[:6]


def summary(rows, names):
    reports = {}
    for cohort in ("hard_screened", "true_miss_on_verification_tape"):
        matched = []
        for row in rows:
            trials = [t["arms"] for t in row["outcomes"]
                      if cohort == "hard_screened" or t["mpc_missing_intervals"]]
            if trials:
                matched.append({"family": row["family"], "seed": row["seed"], "verification": trials})
        if not matched:
            continue
        stats = bands(matched, names, STATS_FIELDS)
        for family, report in stats.items():
            cases = [row for row in matched if family in ("pooled", row["family"])]
            report.pop("replicas_per_world")
            trials = [t for row in cases for t in row["verification"]]
            report["tapes"] = len(trials)
            report["all_baseline_gaps_recaptured"] = {
                name: sum(t[name]["captured_all_baseline_gaps"] for t in trials) for name in names}
            for arm, checks in report["contrast_checks"].items():
                effects = report["vs_reference"][arm]
                checks.pop("execution_within_upper_margins")
                checks.pop("execution_equivalent")
                checks["scan_time_within_upper_margin"] = effects["seconds"]["ci95"][1] <= .10
        reports[cohort] = stats
    return reports


def run_discovery(args):
    cases = read_rows(args.out / "development_hard_cases.jsonl.gz")
    rows = run_parallel(discover, cases, args.jobs, "hard-state privileged search")
    save_archive(args.out / "development_search.jsonl.gz", rows)
    requests = {source: ordered_nominations(rows, source) for source in SOURCES}
    count = min(map(len, requests.values()))
    requests = {source: values[:count] for source, values in requests.items()}
    stats = summary(rows, DEV_ARMS)
    report = {"search_proposals": {source: sum(r["discovery"][source]["proposals"] for r in rows) for source in SOURCES},
              "admissible": {source: sum(r["discovery"][source]["admissible"] for r in rows) for source in SOURCES},
              "admissible_with_caps": {source: sum(r["discovery"][source]["admissible_with_caps"] for r in rows)
                                       for source in SOURCES},
              "used_proposal": {source: sum(r["discovery"][source]["used_proposal"] for r in rows) for source in SOURCES},
              "requests": requests, "paired": stats,
              "cem_vs_random": summary(rows, ("privileged_random", "privileged_cem"))}
    write_json(args.out / "development_search_summary.json", report)


def verify(job):
    row, requests = job
    family, seed = row["family"], row["seed"]
    c, scene, root, prefix, stock, incumbent = stock_options(family, seed)
    scenarios = posterior_worlds(root, PUBLIC_ROLLOUTS, 781_849)
    forecast = rollout_scores(c, root, stock, scenarios)
    reference = forecast[list(stock).index(incumbent)]
    paths = {"mpc": stock[incumbent], "fixed_revisit": stock.get("revisit", stock[incumbent])}
    for source, extra in (("stock", []), *requests.items()):
        additions = {}
        for i, request in enumerate(extra):
            name = f"extra_{i}"
            candidate = c.free_candidate(root, name, *request)
            if candidate is not None:
                additions[name] = candidate
        menu = stock | additions
        values = forecast if not additions else np.vstack((forecast, rollout_scores(c, root, additions, scenarios)))
        selected, _, _ = choose_from_deltas(list(menu), incumbent, values-reference, targets=TARGETS,
                                            caps=CAPS_WITHOUT_RELATIVE_MOTION)
        paths[f"public_{source}"] = menu[selected]
    outcomes = execute_paths(c, scene, root, prefix, seed, paths, TEST_TAPE)
    return {"family": family, "seed": seed, "outcomes": outcomes,
            "selected": {name: action.name for name, action in paths.items()}}


def run_verify(args):
    cases = read_rows(args.out / "holdout_hard_cases.jsonl.gz")
    source = args.out / "development_search_summary.json"
    requests = json.loads(source.read_text())["requests"]
    rows = run_parallel(verify, [(r, requests) for r in cases], args.jobs, "held-out hard-miss selection")
    save_archive(args.out / "holdout_cases.jsonl.gz", rows)
    report = {"development_sha256": fingerprint(source), "paired": summary(rows, ARMS),
              "cem_vs_stock": summary(rows, ("public_stock", "public_cem")),
              "cem_vs_random": summary(rows, ("public_random", "public_cem")),
              "cem_vs_fixed_revisit": summary(rows, ("fixed_revisit", "public_cem")),
              "used_extra": {name: sum(r["selected"][name].startswith("extra_") for r in rows)
                             for name in ("public_random", "public_cem")}}
    write_json(args.out / "holdout_summary.json", report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("screen", "discover", "verify"))
    p.add_argument("--source", type=Path, default=Path("results/visible_atlas_v1"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=16)
    args = p.parse_args()
    if args.jobs < 1:
        p.error("--jobs must be positive")
    library = args.out / "stock_motion_config"
    library.mkdir(exist_ok=True, parents=True)
    archive = json.loads((args.source / "library.json").read_text())
    write_json(library / "library.json", archive | {"requests": [], "shadow_scale": SHADOW_SCALE})
    copyfile(args.source / "prodmp_fit.npy", library / "prodmp_fit.npy")
    configure(library, controller_type=stock_controller)
    {"screen": run_screen, "discover": run_discovery, "verify": run_verify}[args.phase](args)


if __name__ == "__main__":
    main()
