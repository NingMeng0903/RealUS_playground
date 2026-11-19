"""Test paired training labels at real post-loss and revisit checkpoints."""

import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path
from time import process_time

import joblib
import numpy as np

import benchmark_arbitrary_start as scans
from adaptive_controller import FRAMES_PER_BLOCK, IDENTITY_ACCEPT, MAX_TRAVEL, station_bin
from benchmark_first_interval_data import early_cell
from benchmark_window_encoding import choice_from_predictions, decision_inputs
from diagnose_b_selection import run_parallel, save_archive
from learned_recovery import OBJECTIVE, VISIBLE_TARGETS, fit_targets, outcome_vector, predict_deltas
from run_adaptive_recovery import paired_statistics, real_trial
from run_learned_recovery import CONTROLLERS, FAMILIES, fingerprint, read_rows, write_json
from train_startup_b import actual_values, load_controllers
from window_embedding import ViewCalibration


AFTER_LOSS = "after_loss"
REVISIT = "revisit"
CONTEXTS = (AFTER_LOSS, REVISIT)
CELLS = [(family, context, blocks) for family, context in product(FAMILIES, CONTEXTS)
         for blocks in ((2, 3, 4) if context == AFTER_LOSS else (1, 2, 3))]
SPLITS = {
    "targeted_train": (65_000_000, "ambiguous", 6),
    "unconditioned_train": (65_100_000, "any", 6),
    "stress_test": (65_200_000, "ambiguous", 8),
    "clear_test": (65_300_000, "clear", 4),
}
GUARD_SEED = 65_400_000
LABEL_PURPOSE, AUDIT_PURPOSE, TEST_PURPOSE = 19601, 19602, 19603
AUDIT_REPEATS = 16
PLAUSIBLE = .20
ARMS = ("mpc", "previous_b", "unconditioned_b", "targeted_b")
FIELDS = (*scans.FIELDS, "fresh_query_confirmed", "first_block_true_image", "feedback_revisions")
MODELS = {}


def moving_start(family, context, blocks, seed):
    row = scans.make_row(family, "noise", "rolling", blocks, seed)
    if context == AFTER_LOSS:
        rng = np.random.default_rng([seed, 19610])
        end = row["handoff_frame"] - int(rng.integers(1, 4)) + 1
        row.update(regime="noise_and_loss", drop_frames=[end - int(rng.integers(4, 8)), end])
    return row | {"context": context}


def return_probe(controller, root, prefix):
    """Use only public evidence and existing feasible motions for the common revisit."""
    here = station_bin(root.s)
    observed = {station_bin(p["s"]) for p in prefix.packets if p["y"]}
    options = {}
    for name, action in controller.options_shared(root, renderer="fit").items():
        destination = station_bin(action.s[-1])
        if 0 <= destination < here and destination in observed:
            options[name] = action
    missing = {name: a for name, a in options.items() if station_bin(a.s[-1]) in root.uncovered()}
    return controller.mpc(root, missing or options, False, active=True, informative=True) if options else None


def describe_checkpoint(packets, retractions, cutoff, first_return, focus=None):
    """Classify cases from the actual public packets and evidence ledger."""
    now = packets[-1]
    prior_certificates = packets[cutoff - 1]["visible_ledger"]
    revoked = [e for e in retractions
               if e["frame"] >= first_return and e["posterior"] < IDENTITY_ACCEPT
               and (focus is None or e["bin"] == focus)
               and [e["witness"], e["slot"]] == prior_certificates[e["bin"]]]
    scores = sorted(now["association"], reverse=True)
    strongest = scores[0]
    context_class = "weak"
    if revoked or PLAUSIBLE <= strongest < IDENTITY_ACCEPT:
        context_class = "ambiguous"
    elif strongest >= IDENTITY_ACCEPT:
        context_class = "clear"
    return {"class": context_class, "top_association": strongest,
            "second_association": scores[1] if len(scores) > 1 else 0.,
            "n_candidates": len(scores), "conflicting_bins": sorted({e["bin"] for e in revoked}),
            "first_return_frame": first_return}


def moving_case(row):
    controller, scene, root, prefix = scans.start_case(row, "in_progress")
    if root.frame != row["handoff_frame"] or not controller.can_continue(root):
        return "prefix_unavailable", None
    if row["context"] == AFTER_LOSS:
        start, end = row["drop_frames"]
        history = [p for p in prefix.packets[:start] if p["accepted_slot"] >= 0]
        if not history:
            return "no_prior_confirmation", None
        if any(p["y"] for p in prefix.packets[start:end]):
            raise ValueError("programmed consecutive loss did not occur")
        resumed = [p for p in prefix.packets[end:] if p["y"]]
        cutoff, focus, probe = start, None, None
    else:
        cutoff = root.frame + 1
        probe = return_probe(controller, root, prefix)
        if probe is None:
            return "no_feasible_return", None
        controller.execute(scene, root, prefix, probe)
        focus = station_bin(root.s)
        resumed = [p for p in prefix.packets[cutoff:] if p["y"] and station_bin(p["s"]) == focus]
        history = [p for p in prefix.packets[:cutoff] if p["y"] and station_bin(p["s"]) == focus]
        if not history:
            raise ValueError("revisit has no earlier image in the same interval")
    last = prefix.packets[-1]
    if not resumed or not last["y"]:
        return "no_packet_at_checkpoint", None
    if not controller.can_continue(root) or not controller.options_shared(root, renderer="fit"):
        return "no_feasible_continuation", None

    public = describe_checkpoint(prefix.packets, prefix.retractions, cutoff, resumed[0]["frame"], focus)
    public.update(frame=root.frame, station_mm=root.s, lateral_mm=root.x, previously_seen_frame=history[-1]["frame"],
                  return_action=None if probe is None else probe.name)
    evaluator = {
        "target_seen_before": any(0 in p["roles_truth"] for p in history),
        "target_in_recovered_packets": any(0 in p["roles_truth"] for p in resumed),
        "target_in_checkpoint": 0 in last["roles_truth"],
        "wrong_at_checkpoint": root.slot >= 0 and last["roles_truth"][root.slot] != 0,
        "active_wrong_certificate": any(
            w is not None and prefix.packets[w.frame]["roles_truth"][w.slot] != 0 for w in root.ledger),
    }
    return public["class"], (controller, scene, root, prefix, public, evaluator)


def screen_cell(job):
    family, context, blocks, seed0, group, quota = job
    selected, counts = [], Counter()
    for offset in range(1000):
        row = moving_start(family, context, blocks, seed0 + offset)
        tag, case = moving_case(row)
        counts[tag] += 1
        if case is None or (group != "any" and tag != group):
            continue
        selected.append(row | {"checkpoint": case[4], "checkpoint_evaluator": case[5]})
        if len(selected) == quota:
            return {"seed": seed0, "counts": dict(counts), "cases": selected}
    raise ValueError(f"insufficient {group} examples in {family}/{context}/{blocks}: {counts}")


def prepare(out, jobs):
    manifests, used = {}, set()
    for name, (seed, group, quota) in SPLITS.items():
        cohorts = run_parallel(screen_cell, [(*cell, seed + 1000 * i, group, quota)
                                             for i, cell in enumerate(CELLS)], jobs, name)
        rows = [row for cohort in cohorts for row in cohort["cases"]]
        save_split(out, name, rows, manifests, used)
        manifests[name]["screening"] = [dict(cell=cell, **{k: v for k, v in cohort.items() if k != "cases"})
                                        for cell, cohort in zip(CELLS, cohorts, strict=True)]

    early = list(product(FAMILIES, scans.REGIMES, (False, True), (False, True)))
    cohorts = run_parallel(early_cell, [(*cell, GUARD_SEED + 1000 * i, 4)
                                        for i, cell in enumerate(early)], jobs, "first-interval guard")
    save_split(out, "first_interval_test", [row for c in cohorts for row in c["cases"]], manifests, used)
    later = list(product(FAMILIES, scans.REGIMES, ("cold",), range(1, scans.BINS)))
    cohorts = run_parallel(scans.sample_cell, [(*cell, GUARD_SEED + 20_000 + 1000 * i, 2)
                                                for i, cell in enumerate(later)], jobs, "later-start guard")
    save_split(out, "later_test", [row for c in cohorts for row in c["cases"]], manifests, used)
    write_json(out / "splits.json", manifests)


def save_split(out, name, rows, manifests, used):
    seeds = {r["seed"] for r in rows}
    if len(seeds) != len(rows) or seeds & used:
        raise ValueError("scene seeds overlap")
    used.update(seeds)
    path = out / f"{name}_starts.jsonl.gz"
    save_archive(path, rows)
    manifests[name] = {"scenes": len(rows), "sha256": fingerprint(path)}


def decision_case(row, task):
    if "context" not in row:
        return scans.decision_row(row, task)
    _, case = moving_case(row)
    if case is None or case[4] != row["checkpoint"] or case[5] != row["checkpoint_evaluator"]:
        raise ValueError(f"checkpoint no longer matches the frozen prefix for {row['seed']}")
    controller, scene, root, prefix, _, _ = case
    menu, inputs, mpc_cpu, timings = decision_inputs(controller, root, prefix, row, scans.CALIBRATION)
    inputs.update(task=task, first_required_bin=root.first_required_bin, handoff_s=root.s, handoff_x=root.x,
                  pending_intervals=root.uncovered())
    return controller, scene, root, prefix, menu, inputs, mpc_cpu + timings["calibrated"], mpc_cpu


def trial(controller, root, prefix, future, action, drop_frames):
    result, record = real_trial(controller, root, prefix, future, [action])
    if result["max_handoff_error"] >= 1e-7 or result["tcp_mm"] > MAX_TRAVEL + 1e-7:
        raise ValueError("execution exceeded continuity or travel limits")
    if drop_frames and any(p["y"] for p in record.packets[drop_frames[0]:drop_frames[1]]):
        raise ValueError("a packet arrived in the forced outage")
    physically_imaged = {station_bin(p["s"]) for p in record.packets if 0 in p["roles_truth"]}
    return {k: result[k] for k in (*VISIBLE_TARGETS, "frames", "centered_complete",
                                  "fresh_query_confirmed", "first_block_true_image", "feedback_revisions")} | {
        "endpoint_reached": record.motion[-1][1] >= scans.END - .28,
        "all_task_intervals_imaged": set(range(root.first_required_bin, scans.BINS)) <= physically_imaged,
        "new_target_image": any(0 in p["roles_truth"] for p in record.packets[root.frame + 1:]),
    }


def collect_case(row):
    controller, scene, root, prefix, menu, inputs, _, _ = decision_case(row, "in_progress")
    label_bank = scans.paired_futures(scene, row, scans.LABEL_REPEATS, LABEL_PURPOSE)
    labels = [[outcome_vector(trial(controller, root, prefix, f, action, row["drop_frames"]), VISIBLE_TARGETS)
               for f in label_bank] for action in menu.values()]
    averages = np.mean(labels, axis=1)
    scores = averages @ OBJECTIVE
    incumbent = inputs["names"].index(inputs["incumbent"])
    best = max(range(len(menu)), key=lambda i: (averages[i][0], scores[i], i == incumbent))
    risky = max(range(len(menu)), key=lambda i: (averages[i][2], averages[i][1], -averages[i][0]))
    choices = {"mpc": inputs["incumbent"], "best_on_labels": inputs["names"][best],
               "risky_on_labels": inputs["names"][risky]}
    verification = []
    for future in scans.paired_futures(scene, row, AUDIT_REPEATS, AUDIT_PURPOSE):
        cache = {a: trial(controller, root, prefix, future, menu[a], row["drop_frames"])
                 for a in dict.fromkeys(choices.values())}
        verification.append({arm: cache[a] for arm, a in choices.items()})
    return inputs | {"actual_rollouts": labels, "audit_selection": choices, "verification": verification,
                     "prefix_packets": [{k: v for k, v in p.items() if k not in ("roles_truth", "target_truth")}
                                        for p in prefix.packets]}


def audit_summary(rows):
    groups = {"all": rows} | {context: [r for r in rows if r["context"] == context] for context in CONTEXTS}
    report = {}
    for group, cohort in groups.items():
        contrasts, vs_mpc = [], []
        for row in cohort:
            counts = {arm: np.sum([outcome_vector(t[arm], VISIBLE_TARGETS) for t in row["verification"]], axis=0)
                      for arm in row["audit_selection"]}
            good, risky, mpc = (counts[a] for a in ("best_on_labels", "risky_on_labels", "mpc"))
            if good[0] >= risky[0] + 2 and good[2] <= risky[2] - 2 and good[1] <= risky[1]:
                contrasts.append(row["seed"])
            if good[0] >= mpc[0] + 2 and good[1] <= mpc[1]:
                vs_mpc.append(row["seed"])
        report[group] = {
            "scenes": len(cohort), "contrast_seeds": contrasts, "improvement_over_mpc_seeds": vs_mpc,
            "checkpoint_types": dict(Counter(r["checkpoint"]["class"] for r in cohort)),
            "paired": paired_statistics(cohort, ("mpc", "best_on_labels", "risky_on_labels"),
                                        FIELDS, lambda r: r["verification"]),
        }
    return report


def collect(out, jobs, manifests):
    report = {}
    for name in ("targeted_train", "unconditioned_train"):
        rows = run_parallel(collect_case, checked_starts(out, manifests, name), jobs, f"{name} paired labels and audit")
        archive = out / f"{name}.jsonl.gz"
        save_archive(archive, rows)
        candidates = sum(len(r["names"]) for r in rows)
        report[name] = {"sha256": fingerprint(archive), "states": len(rows), "candidate_rows": candidates,
                        "training_rollouts": candidates * scans.LABEL_REPEATS, "audit_repeats": AUDIT_REPEATS,
                        "audit_rollouts": AUDIT_REPEATS * sum(len(set(r["audit_selection"].values())) for r in rows),
                        "audit": audit_summary(rows)}
    write_json(out / "label_summary.json", report)


def previous_audit_case(row):
    action, _ = choice_from_predictions(row, predict_deltas(MODELS["previous_b"], scans.retrained_inputs(row)))
    existing = next((arm for arm, name in row["audit_selection"].items() if name == action), None)
    if existing is None:
        controller, scene, root, prefix, menu, _, _, _ = decision_case(row, "in_progress")
        outcomes = [trial(controller, root, prefix, f, menu[action], row["drop_frames"])
                    for f in scans.paired_futures(scene, row, AUDIT_REPEATS, AUDIT_PURPOSE)]
    else:
        outcomes = [replica[existing] for replica in row["verification"]]
    verification = [{"mpc": refs["mpc"], "previous_b": result, "best_on_labels": refs["best_on_labels"]}
                    for refs, result in zip(row["verification"], outcomes, strict=True)]
    return {k: row[k] for k in ("seed", "context", "family")} | {
        "actions": {"mpc": row["incumbent"], "previous_b": action, "best_on_labels": row["audit_selection"]["best_on_labels"]},
        "verification": verification,
    }


def audit_previous(out, jobs):
    specs = json.loads((out / "label_summary.json").read_text())
    all_results, report = [], {}
    arms = ("mpc", "previous_b", "best_on_labels")
    for split in ("targeted_train", "unconditioned_train"):
        path = out / f"{split}.jsonl.gz"
        check_file(path, specs[split]["sha256"])
        rows = run_parallel(previous_audit_case, read_rows(path), jobs, f"{split} previous B replay")
        all_results.extend(r | {"split": split} for r in rows)
        report[split] = {}
        for context in CONTEXTS:
            cohort = [r for r in rows if r["context"] == context]
            totals = np.array([[np.sum([outcome_vector(t[a], VISIBLE_TARGETS[:3]) for t in r["verification"]], axis=0)
                                for a in arms] for r in cohort])
            mpc, old, nominee = totals.transpose(1, 0, 2)
            opportunities = {}
            for name, reference in (("mpc", mpc), ("previous_b", old)):
                improves = (nominee[:, 0] >= reference[:, 0] + 2) & (nominee[:, 1] <= reference[:, 1])
                opportunities[name] = [r["seed"] for r, ok in zip(cohort, improves, strict=True) if ok]
            report[split][context] = {"scenes": len(cohort), "opportunities": opportunities,
                                      "outcome_counts": dict(zip(arms, totals.sum(axis=0).astype(int), strict=True))}
    path = out / "previous_b_label_audit.jsonl.gz"
    save_archive(path, all_results)
    write_json(out / "previous_b_label_audit.json", {"archive_sha256": fingerprint(path), "groups": report})


def train(out, original, digests):
    report = json.loads((out / "label_summary.json").read_text())
    models = {}
    for label in ("targeted", "unconditioned"):
        path = out / f"{label}_train.jsonl.gz"
        check_file(path, report[f"{label}_train"]["sha256"])
        added = read_rows(path)
        if {r["seed"] for r in original} & {r["seed"] for r in added}:
            raise ValueError("previous and supplemental states overlap")
        pool = original + added
        features = np.concatenate([scans.retrained_inputs(r) for r in pool])
        targets = np.concatenate([actual_values(r) for r in pool])
        models[f"{label}_b"] = fit_targets(features, targets)
    joblib.dump(models, out / "models.joblib", compress=3)
    write_json(out / "frozen.json", {"models_sha256": fingerprint(out / "models.joblib"),
                                    "shared_sha256": digests, "splits_sha256": fingerprint(out / "splits.json"),
                                    "label_summary_sha256": fingerprint(out / "label_summary.json"),
                                    "retained_training_states": len(original)})


def test_case(job):
    group, row = job
    tasks, by_interval = [], {}
    for task in scans.task_names(row):
        controller, scene, root, prefix, menu, inputs, feature_cpu, mpc_cpu = decision_case(row, task)
        if root.first_required_bin in by_interval:
            tasks.append(by_interval[root.first_required_bin] | {"task": task})
            continue
        choices, cpu = {"mpc": inputs["incumbent"]}, {"mpc": mpc_cpu}
        start = process_time()
        features = scans.retrained_inputs(inputs)
        feature_cpu += process_time() - start
        for arm in ARMS[1:]:
            start = process_time()
            choices[arm], _ = choice_from_predictions(inputs, predict_deltas(MODELS[arm], features))
            cpu[arm] = feature_cpu + process_time() - start
        verification = []
        for future in scans.paired_futures(scene, row, scans.TEST_REPEATS, TEST_PURPOSE):
            cache = {a: trial(controller, root, prefix, future, menu[a], row["drop_frames"])
                     for a in dict.fromkeys(choices.values())}
            verification.append({arm: cache[a] for arm, a in choices.items()})
        result = {k: v for k, v in inputs.items() if k not in ("features", "margins", "calibrated_detection")}
        result.update(group=group, selection=choices, decision_cpu_s=cpu, verification=verification)
        tasks.append(result)
        by_interval[root.first_required_bin] = result
    return {"seed": row["seed"], "tasks": tasks}


def test_summary(rows):
    stress = [r for r in rows if r["group"] == "stress_test"]
    clear = [r for r in rows if r["group"] == "clear_test"]
    first = [r for r in rows if r["group"] == "first_interval_test" and r["task"] == scans.TASKS[0]]
    later = [r for r in rows if r["group"] == "later_test"]
    groups = {"stress": stress, "clear": clear, "first_interval": first} | {
        context: [r for r in stress if r["context"] == context] for context in CONTEXTS}
    groups.update({f"stress_{family}": [r for r in stress if r["family"] == family] for family in FAMILIES})
    groups.update({name: [r for r in later if r["task"] == name] for name in scans.TASKS[:2]})
    report = {}
    fields = ("corrected_complete", "final_wrong", "new_identity_mistake", "fresh_query_confirmed", "new_target_image")
    for name, cohort in groups.items():
        tapes = [t for r in cohort for t in r["verification"]]
        report[name] = {
            "scenes": len(cohort), "runs_per_arm": len(tapes),
            "counts": {a: {k: int(sum(t[a][k] for t in tapes)) for k in fields} for a in ARMS},
            "vs_mpc": paired_statistics(cohort, ARMS, FIELDS, lambda r: r["verification"]),
            "vs_previous": paired_statistics(cohort, ARMS[1:], FIELDS, lambda r: r["verification"]),
            "vs_unconditioned": paired_statistics(cohort, ARMS[2:], FIELDS, lambda r: r["verification"]),
            "mean_decision_cpu_ms": {a: 1000. * float(np.mean([r["decision_cpu_s"][a] for r in cohort])) for a in ARMS},
            "choices": {a: dict(Counter(r["selection"][a] for r in cohort)) for a in ARMS},
        }
    return report


def check_file(path, digest):
    if fingerprint(path) != digest:
        raise ValueError(f"frozen input changed: {path}")


def checked_starts(out, manifests, name):
    path = out / f"{name}_starts.jsonl.gz"
    check_file(path, manifests[name]["sha256"])
    return read_rows(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "collect", "train", "test", "audit_previous"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    out = args.out
    library = Path("results/startup_b_new_library_v1")
    window = Path("results/window_encoding_v1")
    multi = Path("results/arbitrary_start_v1")
    early = Path("results/first_interval_data_v1")
    load_controllers(library)
    previous = json.loads((multi / "frozen.json").read_text())
    supplemented = json.loads((early / "frozen.json").read_text())
    digests = previous["source_sha256"]
    for directory, names in ((library, ("library.json", "prodmp_fit.npy")),
                             (window, ("calibration.json",))):
        for name in names:
            check_file(directory / name, digests[name])
    check_file(multi / "training.jsonl.gz", previous["training_sha256"])
    for name in ("models.joblib", "additional_training.jsonl.gz"):
        check_file(early / name, supplemented["frozen_sha256"][name])
    scans.CALIBRATION = ViewCalibration(**json.loads((window / "calibration.json").read_text())["parameters"])
    shared = {"library": digests["library.json"], "prodmp": digests["prodmp_fit.npy"],
              "calibration": digests["calibration.json"], "original_training": previous["training_sha256"],
              "previous_model": supplemented["frozen_sha256"]["models.joblib"],
              "previous_extra_training": supplemented["frozen_sha256"]["additional_training.jsonl.gz"]}
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        prepare(out, args.jobs)
        return
    manifests = json.loads((out / "splits.json").read_text())
    if args.stage == "collect":
        collect(out, args.jobs, manifests)
        return
    if args.stage == "train":
        original = read_rows(multi / "training.jsonl.gz") + read_rows(early / "additional_training.jsonl.gz")
        train(out, original, shared)
        return
    if args.stage == "audit_previous":
        MODELS["previous_b"] = joblib.load(early / "models.joblib")
        audit_previous(out, args.jobs)
        return

    frozen = json.loads((out / "frozen.json").read_text())
    if shared != frozen["shared_sha256"]:
        raise ValueError("shared inputs changed")
    for name in ("models.joblib", "splits.json", "label_summary.json"):
        check_file(out / name, frozen[f"{name.split('.')[0]}_sha256"])
    MODELS.update(joblib.load(out / "models.joblib"), previous_b=joblib.load(early / "models.joblib"))
    jobs = [(group, r) for group in ("stress_test", "clear_test", "first_interval_test", "later_test")
            for r in checked_starts(out, manifests, group)]
    rows = run_parallel(test_case, jobs, args.jobs, "identity-recovery held-out closed loops")
    tasks = [t for r in rows for t in r["tasks"]]
    path = out / "test.jsonl.gz"
    save_archive(path, tasks)
    write_json(out / "summary.json", {"test_sha256": fingerprint(path), "scenes": len(rows), "groups": test_summary(tasks)})


if __name__ == "__main__":
    main()
