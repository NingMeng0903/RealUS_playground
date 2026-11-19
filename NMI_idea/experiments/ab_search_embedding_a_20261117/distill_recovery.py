"""Distill publicly nominated and independently replayed recovery choices."""

import argparse
from collections import Counter
import gzip
import json
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter, process_time

import joblib
import numpy as np

from adaptive_controller import FreeCandidate, station_bin
from adaptive_teacher import posterior_worlds
from budgeted_recovery import CAPS
from distill_policy import VARIANTS, choose_distilled, features_for_choices, fit_distillers, verified_deltas, vocabulary
from learned_recovery import (OBJECTIVE, TARGETS, choose_from_deltas, features_for_menu, outcome_vector,
                              predict_deltas)
from run_adaptive_recovery import (MARGINS, paired_statistics, public_root, real_trial, root_report,
                                   shared_prefix, tape_for, trace)
from run_learned_recovery import (CONTROLLERS, FAMILIES, candidate_set, configure, fingerprint, jsonable,
                                 read_rows, rollout_values, summarize, write_json)


PARALLEL_RANDOM_SEED = 15_210_000
SCOUT_SEED = 15_310_000
TEACHER_SEED = 751_239
VERIFICATION_TAPE = 1861
STRATA = ("past_gap", "coverage_ahead")
DEV_SEED = 16_210_000
TEST_SEED = 17_210_000
TEST_ARMS = ("mpc", "trained_v1", "distilled_b", "online_b", "online_without_extra_cem")
DEV_ARMS = ("mpc", *VARIANTS)
BENCH_MODELS = {}


def scan_stratum(root):
    station, query = station_bin(root.s), root.query_bin()
    if query < station and root.covered.bit_count() == 1:
        return "past_gap"
    if query > station + 1:
        return "coverage_ahead"
    return "ordinary"


def scout(seed):
    controller = CONTROLLERS["parallel"]
    root, _ = shared_prefix(controller, controller.world.draw(seed))
    return seed, scan_stratum(root)


def actions_to_verify(names, incumbent, deltas):
    teacher, deltas, allowed = choose_from_deltas(names, incumbent, deltas)
    order = sorted(range(len(names)), key=lambda i: deltas[i] @ OBJECTIVE, reverse=True)
    nominated = {incumbent: "incumbent"}
    if teacher != incumbent:
        nominated[teacher] = "teacher"
    for eligible, reason in ((True, "eligible_challenger"), (False, "near_rejected_challenger")):
        alternative = next((names[i] for i in order if allowed[i] == eligible and names[i] not in nominated), None)
        if alternative is not None:
            nominated[alternative] = reason
    return teacher, nominated


def collect_case(job):
    family, seed, source, selection, replicas = job
    controller = CONTROLLERS[family]
    scene = controller.world.draw(seed)
    root, prefix = shared_prefix(controller, scene)
    menu, incumbent, features = candidate_set(controller, root, prefix)
    if not menu:
        return {"family": family, "seed": seed, "source": source, "names": []}
    names = list(menu)
    if selection is None:
        bank = posterior_worlds(root, 12, TEACHER_SEED)
        prediction = rollout_values(controller, root, menu, bank).mean(axis=1)
        selection_samples = 12
    else:
        saved_names, saved_incumbent, prediction = selection
        if names != saved_names or incumbent != saved_incumbent:
            raise ValueError("the archive and the live shared controller disagree")
        selection_samples = 10
    deltas = prediction - prediction[names.index(incumbent)]
    teacher, nominees = actions_to_verify(names, incumbent, deltas)
    worlds = [tape_for(scene.latent, seed, k, VERIFICATION_TAPE) for k in range(replicas)]
    verified = {}
    for name in nominees:
        verified[name] = [outcome_vector(real_trial(controller, root, prefix, world, [menu[name]])[0])
                          for world in worlds]
    return {"family": family, "seed": seed, "source": source, "stratum": scan_stratum(root),
            "public_root": public_root(root), "names": names, "incumbent": incumbent, "teacher": teacher,
            "features": features, "selection_samples": selection_samples, "nominees": nominees,
            "posterior_delta": deltas, "verified_outcomes": verified}


def collect(args):
    configure(args.source)
    previous = args.source / "train.jsonl.gz"
    rows = read_rows(previous)
    jobs = []
    for row in rows:
        if row["names"]:
            selection = (row["names"], row["incumbent"], np.mean(row["posterior_rollouts"], axis=1))
            jobs.append((row["family"], row["seed"], "archived_training", selection, args.replicas))

    started = perf_counter()
    gathered = []
    with get_context("fork").Pool(args.jobs) as pool:
        selected_states = Counter()
        for seed, stratum in pool.imap(scout, range(SCOUT_SEED, SCOUT_SEED + args.scout_cases), chunksize=8):
            if stratum in STRATA and selected_states[stratum] < args.stratum_cases:
                selected_states[stratum] += 1
                jobs.append(("parallel", seed, "public_stratum", None, args.replicas))
            if all(selected_states[key] >= args.stratum_cases for key in STRATA):
                break
        for j in range(args.random_cases):
            jobs.append(("parallel", PARALLEL_RANDOM_SEED + j, "new_random", None, args.replicas))
        print(f"teacher collection {len(jobs)} scenes; extra public strata {dict(selected_states)}", flush=True)
        for index, row in enumerate(pool.imap_unordered(collect_case, jobs), start=1):
            gathered.append(row)
            if index % 24 == 0 or index == len(jobs):
                print(f"teacher {index}/{len(jobs)}", flush=True)
    archive = args.out / "teacher_train.jsonl.gz"
    pending = archive.with_suffix(".tmp")
    with gzip.open(pending, "wt", encoding="utf-8", compresslevel=3) as stream:
        for row in sorted(gathered, key=lambda r: r["seed"]):
            stream.write(json.dumps(row, default=jsonable, allow_nan=False, separators=(",", ":")) + "\n")
    pending.replace(archive)
    valid = [row for row in gathered if row["names"]]
    write_json(args.out / "teacher_data.json", {
        "scenes": len(gathered), "labeled": len(valid), "scenes_by_source": Counter(r["source"] for r in valid),
        "scenes_by_family": Counter(r["family"] for r in valid), "scouted_strata": selected_states,
        "tapes_per_candidate": args.replicas, "nominees": sum(len(r["nominees"]) for r in valid),
        "elapsed_s": perf_counter() - started, "archive_sha256": fingerprint(archive),
        "v1_training_sha256": fingerprint(previous),
        "library_sha256": fingerprint(args.source / "library.json"),
        "prodmp_sha256": fingerprint(args.source / "prodmp_fit.npy"),
    })


def run_benchmark_case(job):
    family, seed, replicas, split = job
    controller = CONTROLLERS[family]
    scene = controller.world.draw(seed)
    root, prefix = shared_prefix(controller, scene)
    started = process_time()
    menu = controller.options_shared(root, renderer="fit")
    incumbent = controller.mpc(root, menu, False, active=True, informative=True).name if menu else None
    shared_cpu = process_time() - started
    arms = TEST_ARMS if split == "test" else DEV_ARMS
    chosen = dict.fromkeys(arms, incumbent)
    timing = dict.fromkeys(arms, shared_cpu)
    predictions = {}
    if menu:
        names = list(menu)
        started = process_time()
        inputs = features_for_menu(controller, root, menu, incumbent, prefix.packets[-1]["association"])
        features_cpu = process_time() - started
        started = process_time()
        matrix = features_for_choices(inputs, names, incumbent, root.covered, root.s,
                                      root.query_bin(), BENCH_MODELS["actions"])
        features_distilled_cpu = features_cpu + process_time() - started

        variants = {"distilled_b": BENCH_MODELS["selected"]} if split == "test" else dict(zip(VARIANTS, VARIANTS))
        for arm, variant in variants.items():
            started = process_time()
            chosen[arm], scores = choose_distilled(variant, BENCH_MODELS["variants"][variant],
                                                  matrix, names, incumbent)
            timing[arm] = shared_cpu + features_distilled_cpu + process_time() - started
            predictions[arm] = {"scores": scores}

        if split == "test":
            started = process_time()
            predictions_v1 = predict_deltas(BENCH_MODELS["trained_v1"], inputs)
            chosen["trained_v1"], deltas, allowed = choose_from_deltas(names, incumbent, predictions_v1)
            timing["trained_v1"] = shared_cpu + features_cpu + process_time() - started
            predictions["trained_v1"] = {"delta": deltas, "eligible": allowed}

            started = process_time()
            bank = posterior_worlds(root, 12, TEACHER_SEED)
            predicted = rollout_values(controller, root, menu, bank).mean(axis=1)
            deltas = predicted - predicted[names.index(incumbent)]
            chosen["online_b"], values, allowed = choose_from_deltas(names, incumbent, deltas)
            timing["online_b"] = shared_cpu + process_time() - started
            predictions["online_b"] = {"delta": values, "eligible": allowed}
            reduced = [i for i, name in enumerate(names) if not name.startswith("cem_") or name == incumbent]
            chosen["online_without_extra_cem"], _, _ = choose_from_deltas([names[i] for i in reduced],
                                                                         incumbent, deltas[reduced])
            timing["online_without_extra_cem"] = timing["online_b"]

    trials = []
    for replica in range(replicas):
        scene_after = tape_for(scene.latent, seed, replica, 2161 if split == "test" else 2160)
        executed, outcomes, logs = {}, {}, {}
        detailed = split == "test" and replica == 0
        for name in dict.fromkeys(chosen.values()):
            result, record = real_trial(controller, root, prefix, scene_after, [menu.get(name)], detailed=detailed)
            if result["max_handoff_error"] > 1e-7:
                raise ValueError("position or velocity changed discontinuously")
            executed[name] = result, trace(record, after_frame=root.frame) if detailed else None
        for arm, name in chosen.items():
            result, log = executed[name]
            outcomes[arm] = result | {"first_decision_cpu_s": timing[arm],
                                      "seconds_plus_first_decision_cpu": result["seconds"] + timing[arm]}
            if detailed:
                logs[arm] = log
        trials.append({"replica": replica, "outcomes": outcomes, "traces": logs})
    return {
        "family": family, "seed": seed, "stratum": scan_stratum(root),
        "checkpoint": root_report(root, prefix), "public_root": public_root(root), "names": list(menu),
        "prefix": trace(prefix) if split == "test" else None, "selected": chosen, "predictions": predictions,
        "renderers": {name: act.renderer if isinstance(act, FreeCandidate) else "legacy_prodmp"
                      for name, act in menu.items()},
        "verification": trials,
    }


def benchmark(args, split):
    base = TEST_SEED if split == "test" else DEV_SEED
    jobs = [(family, base + 100_000 * i + args.offset + j, args.replicas, split)
            for i, family in enumerate(FAMILIES) for j in range(args.cases)]
    started = perf_counter()
    gathered = []
    with get_context("fork").Pool(min(args.jobs, len(jobs))) as pool:
        for index, row in enumerate(pool.imap_unordered(run_benchmark_case, jobs), start=1):
            gathered.append(row)
            if index % 16 == 0 or index == len(jobs):
                print(f"{split} {index}/{len(jobs)}", flush=True)
    archive = args.out / f"{split}.jsonl.gz"
    pending = archive.with_suffix(".tmp")
    gathered.sort(key=lambda r: r["seed"])
    with gzip.open(pending, "wt", encoding="utf-8", compresslevel=3) as stream:
        for row in gathered:
            stream.write(json.dumps(row, default=jsonable, allow_nan=False, separators=(",", ":")) + "\n")
    pending.replace(archive)
    report = summarize(gathered, TEST_ARMS if split == "test" else DEV_ARMS)
    report["elapsed_s"] = perf_counter() - started
    return gathered, report


def label_counts(rows):
    counts = {}
    for family in FAMILIES:
        family_counts = Counter()
        for row in rows:
            if row["family"] != family or not row["names"]:
                continue
            deltas = verified_deltas(row)
            family_counts["states"] += 1
            family_counts["actual_positive_teacher"] += int(deltas[row["teacher"]][0][0] > 0.)
            family_counts["verified_teacher"] += int(deltas[row["teacher"]][1])
            family_counts["any_verified"] += int(any(verified for _, verified in deltas.values()))
            family_counts["alternatives_checked"] += len(deltas) - 1
        counts[family] = family_counts
    return counts


def train(args):
    configure(args.source)
    teacher_path = args.out / "teacher_train.jsonl.gz"
    rows = read_rows(teacher_path)
    actions = vocabulary(rows)
    BENCH_MODELS.update(actions=actions, variants=fit_distillers(rows, actions))
    independent, dev = benchmark(args, "dev")
    train_seeds = {r["seed"] for r in rows}
    if train_seeds & {r["seed"] for r in independent} or max(r["seed"] for r in independent) >= TEST_SEED:
        raise ValueError("training, development and test must be disjoint")

    def order(variant):
        results = dev["pooled"]["vs_reference"][variant]
        deltas = {metric: item["delta"] for metric, item in results.items()}
        deltas["seconds"] = deltas["seconds_plus_first_decision_cpu"]
        violation = sum(max(0., deltas[metric] - cap) / cap for metric, cap in CAPS.items())
        return violation, -deltas["corrected_complete"]

    chosen = min(VARIANTS, key=order)
    BENCH_MODELS["selected"] = chosen
    joblib.dump(BENCH_MODELS, args.out / "distilled.joblib", compress=3)
    write_json(args.out / "dev_summary.json", dev)
    write_json(args.out / "freeze.json", {
        "selected": chosen, "selection_order": {mode: order(mode) for mode in VARIANTS},
        "teacher_training": label_counts(rows), "train_sha256": fingerprint(teacher_path),
        "dev_sha256": fingerprint(args.out / "dev.jsonl.gz"),
        "student_sha256": fingerprint(args.out / "distilled.joblib"),
        "library_sha256": fingerprint(args.source / "library.json"),
        "prodmp_sha256": fingerprint(args.source / "prodmp_fit.npy"),
        "v1_sha256": fingerprint(args.source / "models.joblib"),
    })
    print(f"frozen {chosen}; labels {label_counts(rows)}; development {dict((m, order(m)) for m in VARIANTS)}", flush=True)


def test(args):
    configure(args.source)
    frozen = json.loads((args.out / "freeze.json").read_text())
    for file, key in ((args.out / "distilled.joblib", "student"), (args.source / "library.json", "library"),
                      (args.source / "prodmp_fit.npy", "prodmp"), (args.source / "models.joblib", "v1")):
        if fingerprint(file) != frozen[key + "_sha256"]:
            raise ValueError(f"frozen artifact mismatch: {file}")
    BENCH_MODELS.update(joblib.load(args.out / "distilled.joblib"))
    BENCH_MODELS["trained_v1"] = joblib.load(args.source / "models.joblib")["learned_b"]
    cases, report = benchmark(args, "test")
    for family in (*FAMILIES, "pooled"):
        rows = [row for row in cases if row["family"] == family or family == "pooled"]
        contrasts = {}
        for pair in (("trained_v1", "distilled_b"), ("online_without_extra_cem", "online_b")):
            result = paired_statistics(rows, pair, TARGETS, lambda r: [t["outcomes"] for t in r["verification"]])
            contrasts[pair[1] + "_vs_" + pair[0]] = result["vs_reference"][pair[1]]
        report[family]["direct_contrasts"] = contrasts
    write_json(args.out / "test_summary.json", report | {
        "selected": frozen["selected"], "margins": MARGINS, "cases_per_family": args.cases,
        "replicas_per_case": args.replicas, "seed_offset": args.offset,
        "freeze_sha256": fingerprint(args.out / "freeze.json"),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--random-cases", type=int, default=48)
    collect_parser.add_argument("--stratum-cases", type=int, default=40)
    collect_parser.add_argument("--scout-cases", type=int, default=2500)
    train_parser = subparsers.add_parser("train")
    test_parser = subparsers.add_parser("test")
    for sub in (collect_parser, train_parser, test_parser):
        sub.add_argument("--replicas", type=int, default=12 if sub is collect_parser else 6 if sub is train_parser else 4)
        sub.add_argument("--jobs", type=int, default=24)
    for sub in (train_parser, test_parser):
        sub.add_argument("--cases", type=int, default=64 if sub is train_parser else 96)
        sub.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if min(args.replicas, args.jobs) <= 0:
        parser.error("positive worker and replica counts required")
    if args.phase == "collect":
        if min(args.stratum_cases, args.random_cases, args.scout_cases) < 0:
            parser.error("negative scene count")
    elif args.cases < 1 or args.offset < 0 or args.offset + args.cases >= 100_000:
        parser.error("invalid independent scenario namespace")
    args.out.mkdir(parents=True, exist_ok=True)
    {"collect": collect, "train": train, "test": test}[args.phase](args)


if __name__ == "__main__":
    main()
