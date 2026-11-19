"""Cross teacher and learned selection on new, paired development cases."""

import argparse
from collections import Counter
import gzip
import json
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np

from budgeted_recovery import CAPS
from learned_recovery import OBJECTIVE, TARGETS, choose_from_deltas, predict_deltas
from recovery_world import Scene, noise_tape
from run_adaptive_recovery import (OUTCOMES, paired_statistics, public_root, real_trial, root_report,
                                   shared_prefix, tape_for, trace)
from run_learned_recovery import (CONTROLLERS, FAMILIES, candidate_set, configure, fingerprint, jsonable,
                                 read_rows, rollout_values, write_json)


SEED = 18_210_000
MAX_OMITTED_MASS = 1e-4
ARMS = ("mpc", "student", "teacher_filter_student_rank", "student_filter_teacher_rank", "teacher")
FROZEN_MODEL = {}
VERIFY_PURPOSE = 2313


def posterior_bank(root, seed, replicas, purpose):
    """Enumerate significant posterior hypotheses with common nuisance draws."""
    belief = np.asarray(root.belief)
    order = np.argsort(-belief, kind="stable")
    selected = order[:np.searchsorted(np.cumsum(belief[order]), 1. - MAX_OMITTED_MASS) + 1]
    mass = float(belief[selected].sum())
    rng = np.random.default_rng([seed, purpose])
    disturbances = [noise_tape(rng) for _ in range(replicas)]
    worlds = [Scene(int(latent), tape) for latent in selected for tape in disturbances]
    weights = np.repeat(belief[selected] / mass / replicas, replicas)
    return worlds, weights, selected, max(0., 1. - mass)


def masked_best(names, incumbent, allowed, objective):
    selected = max(np.flatnonzero(allowed), key=lambda i: (objective[i], names[i] == incumbent))
    return names[selected]


def cross_choices(names, incumbent, teacher_delta, student_delta):
    _, teacher, teacher_allowed = choose_from_deltas(names, incumbent, teacher_delta)
    _, student, student_allowed = choose_from_deltas(names, incumbent, student_delta)
    teacher_order, student_order = teacher @ OBJECTIVE, student @ OBJECTIVE
    selected = {
        "mpc": incumbent,
        "student": masked_best(names, incumbent, student_allowed, student_order),
        "teacher_filter_student_rank": masked_best(names, incumbent, teacher_allowed, student_order),
        "student_filter_teacher_rank": masked_best(names, incumbent, student_allowed, teacher_order),
        "teacher": masked_best(names, incumbent, teacher_allowed, teacher_order),
    }
    return selected, {"delta": teacher, "allowed": teacher_allowed}, {"delta": student, "allowed": student_allowed}


def actual_replays(controller, root, prefix, scene, seed, menu, selected, replicas):
    trials = []
    for replica in range(replicas):
        disturbance = tape_for(scene.latent, seed, replica, VERIFY_PURPOSE)
        outcomes, traces = {}, {}
        for action in dict.fromkeys(selected.values()):
            metrics, record = real_trial(controller, root, prefix, disturbance, [menu.get(action)],
                                         detailed=replica == 0)
            if metrics["max_handoff_error"] > 1e-7:
                raise ValueError("discontinuous segment handoff")
            outcomes[action] = metrics
            if replica == 0:
                traces[action] = trace(record, after_frame=root.frame)
        trials.append({"replica": replica, "outcomes": {arm: outcomes[name] for arm, name in selected.items()},
                       "traces": {arm: traces[name] for arm, name in selected.items()} if replica == 0 else {}})
    return trials


def diagnose_case(job):
    family, seed, samples, replicas = job
    controller = CONTROLLERS[family]
    scene = controller.world.draw(seed)
    root, prefix = shared_prefix(controller, scene)
    menu, incumbent, features = candidate_set(controller, root, prefix)
    names = list(menu)
    if menu:
        worlds, weights, support, omitted = posterior_bank(root, seed, samples, 2311)
        table = np.einsum("amk,m->ak", rollout_values(controller, root, menu, worlds), weights)
        reference = table[names.index(incumbent)]
        selected, teacher, student = cross_choices(names, incumbent, table - reference,
                                                   predict_deltas(FROZEN_MODEL["learned_b"], features))

        # Independent evaluation of the selected actions plus up to two
        # other teacher-eligible actions close to the top of its ranking.
        ordered = np.argsort(-(teacher["delta"] @ OBJECTIVE), kind="stable")
        runners_up = [names[i] for i in ordered if teacher["allowed"][i] and names[i] not in selected.values()][:2]
        check_names = list(dict.fromkeys([*selected.values(), *runners_up]))
        check_menu = {name: menu[name] for name in check_names}
        worlds, weights, _, _ = posterior_bank(root, seed, samples, 2312)
        check = np.einsum("amk,m->ak", rollout_values(controller, root, check_menu, worlds), weights)
        independent_delta = check - check[check_names.index(incumbent)]
        independent_choice, independent_delta, independent_allowed = choose_from_deltas(
            check_names, incumbent, independent_delta)
        validation = {"names": check_names, "delta": independent_delta, "allowed": independent_allowed,
                      "best_among_checked": independent_choice}
    else:
        selected = dict.fromkeys(ARMS)
        teacher, student, validation, support, omitted = None, None, None, [], 0.

    return {"family": family, "seed": seed, "checkpoint": root_report(root, prefix),
            "public_root": public_root(root), "prefix": trace(prefix), "names": names, "selected": selected,
            "teacher": teacher, "student": student, "independent_forecast": validation,
            "posterior_support": support, "omitted_mass": omitted,
            "verification": actual_replays(controller, root, prefix, scene, seed, menu, selected, replicas)}


def trial_outcomes(row):
    return [trial["outcomes"] for trial in row["verification"]]


def diagnostics(cases):
    changes = {a: sum(r["selected"][a] != r["selected"]["student"] for r in cases) for a in ARMS[2:]}
    reference_checked, reference_allowed, checked_same = 0, 0, 0
    blocked = Counter()
    for row in cases:
        actions = row["selected"]
        if not row["names"]:
            continue
        teacher_name = actions["teacher"]
        i = row["names"].index(teacher_name)
        if not row["student"]["allowed"][i]:
            blocked["any"] += 1
            for metric, cap in CAPS.items():
                if row["student"]["delta"][i][TARGETS.index(metric)] > cap + 1e-8:
                    blocked[metric] += 1
        check = row["independent_forecast"]
        reference_checked += 1
        reference_allowed += int(check["allowed"][check["names"].index(teacher_name)])
        checked_same += int(check["best_among_checked"] == teacher_name)
    return {"changes_from_student": changes, "teacher_blocked_by_student": blocked,
            "teacher_actions_checked": reference_checked, "teacher_actions_passing_independent_check": reference_allowed,
            "teacher_actions_still_best_among_checked": checked_same}


def summarize_cross(cases, arms=ARMS):
    report = {}
    cohorts = {family: [r for r in cases if r["family"] == family] for family in FAMILIES}
    cohorts["pooled"] = cases
    for family, rows in cohorts.items():
        if not rows:
            continue
        result = paired_statistics(rows, arms, OUTCOMES, trial_outcomes)
        result["counts"] = {
            arm: {key: round(sum(t[arm][key] for row in rows for t in trial_outcomes(row)))
                  for key in ("corrected_complete", "final_wrong", "new_centered_mistake")}
            for arm in arms
        }
        result["replays_per_arm"] = sum(len(row["verification"]) for row in rows)
        result["compared_with_frozen_student"] = paired_statistics(
            rows, ("student", *(arm for arm in arms if arm not in ("mpc", "student"))), OUTCOMES, trial_outcomes
        )["vs_reference"]
        result["decisions"] = diagnostics(rows)
        report[family] = result
    return report


def run_parallel(function, jobs, workers, name):
    rows = []
    with get_context("fork").Pool(min(workers, len(jobs))) as pool:
        for index, row in enumerate(pool.imap_unordered(function, jobs), 1):
            rows.append(row)
            if index % 8 == 0 or index == len(jobs):
                print(f"{name} {index}/{len(jobs)}", flush=True)
    return sorted(rows, key=lambda row: row["seed"])


def save_archive(path, rows):
    pending = path.with_suffix(".tmp")
    with gzip.open(pending, "wt", encoding="utf-8", compresslevel=3) as stream:
        for row in rows:
            stream.write(json.dumps(row, default=jsonable, allow_nan=False, separators=(",", ":")) + "\n")
    pending.replace(path)


def diagnose(args):
    configure(args.source)
    FROZEN_MODEL.update(joblib.load(args.source / "models.joblib"))
    jobs = [(family, SEED + i * 100_000 + args.offset + j, args.samples, args.replicas)
            for i, family in enumerate(FAMILIES) for j in range(args.cases)]
    started = perf_counter()
    rows = run_parallel(diagnose_case, jobs, args.jobs, "diagnostic")
    archive = args.out / "cross_cases.jsonl.gz"
    save_archive(archive, rows)
    reports = summarize_cross(rows)
    write_json(args.out / "cross_summary.json", reports | {
        "elapsed_s": perf_counter() - started, "samples_per_retained_hypothesis": args.samples,
        "observed_repeats": args.replicas, "omitted_mass_bound": MAX_OMITTED_MASS,
        "pooled_mean_posterior_support": float(np.mean([len(row["posterior_support"]) for row in rows])),
        "max_omitted_mass": max(row["omitted_mass"] for row in rows),
        "frozen_student_sha256": fingerprint(args.source / "models.joblib"),
        "library_sha256": fingerprint(args.source / "library.json"), "archive_sha256": fingerprint(archive),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=32, help="independent development cases per family")
    parser.add_argument("--samples", type=int, default=12, help="disturbance tapes per retained hypothesis")
    parser.add_argument("--replicas", type=int, default=8, help="independent evaluator tapes per case")
    parser.add_argument("--jobs", type=int, default=24)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if min(args.cases, args.samples, args.replicas, args.jobs) <= 0 or not 0 <= args.offset < 100_000 - args.cases:
        parser.error("invalid sampling configuration")
    args.out.mkdir(parents=True, exist_ok=True)
    diagnose(args)


if __name__ == "__main__":
    main()
