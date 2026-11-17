"""Audit acquisition after reranking recovery actions with qualified evidence."""

import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from adaptive_controller import BINS, CENTER_MM, END, QUALIFIED_EVIDENCE, station_bin
from adaptive_teacher import posterior_worlds
from audit_teacher_ceiling import frame_coverage
from diagnose_b_selection import run_parallel, save_archive
from learned_recovery import choose_from_deltas
from run_adaptive_recovery import OUTCOMES, paired_statistics, public_root, real_trial, shared_prefix, tape_for, trace
from run_learned_recovery import CONTROLLERS, FAMILIES, config, configure, fingerprint, rollout_values, write_json


SEED = 19_210_000
VERIFY_PURPOSE = 4163
ARMS = ("mpc", "teacher")
NO_IMAGE = ("unvisited", "off_center_motion", "off_center_exposures", "target_packet_missing")


def interval_failures(controller, scene, record, metrics):
    """Use hidden geometry only to diagnose the path that was really executed."""
    visited = physical = 0
    for _, s, x, _, _, q in record.motion:
        j = station_bin(s)
        if j >= 0:
            bit = 1 << j
            visited |= bit
            if not physical & bit and abs(controller.world.target_grid(s, q)[scene.latent, 0] - x) <= CENTER_MM:
                physical |= bit
    centered, raw, accepted, _ = frame_coverage(record)
    assert physical & centered == centered
    rejected = {(event["witness"], event["bin"]) for event in record.retractions}
    retained = {e["bin"]: e["identity_ok"] and e["center_ok"] for e in record.certificates
                if (e["frame"], e["bin"]) not in rejected}
    return [next((name for mask, name in zip((visited, physical, centered, raw, accepted),
                                            (*NO_IMAGE, "target_unassociated"), strict=True)
                  if not mask & (1 << j)), "qualified" if retained.get(j) and metrics["covered_mask"] & (1 << j)
                 else "evidence_unqualified") for j in range(BINS)]


def paired_case(job):
    family, seed, samples, replicas = job
    controller = CONTROLLERS[family]
    scene = controller.world.draw(seed)
    root, prefix = shared_prefix(controller, scene)
    menu = controller.options_shared(root, renderer="fit")
    selected, values, allowed = dict.fromkeys(ARMS), [], []
    if menu:
        incumbent = controller.mpc(root, menu, False, active=True, informative=True).name
        names = list(menu)
        bank = posterior_worlds(root, samples, 751_239)
        means = rollout_values(controller, root, menu, bank).mean(axis=1)
        winner, values, allowed = choose_from_deltas(names, incumbent, means - means[names.index(incumbent)])
        selected = {"mpc": incumbent, "teacher": winner}

    trials = []
    for replica in range(replicas):
        disturbance = tape_for(scene.latent, seed, replica, VERIFY_PURPOSE)
        executed = {}
        for name in dict.fromkeys(selected.values()):
            metrics, record = real_trial(controller, root, prefix, disturbance, [menu.get(name)],
                                         detailed=replica == 0)
            executed[name] = (metrics | {
                "arrived": record.motion[-1][1] >= END - .28,
                "intervals": interval_failures(controller, scene, record, metrics),
            }, trace(record, after_frame=root.frame) if replica == 0 else None)
        trials.append({"replica": replica, "outcomes": {arm: executed[name][0] for arm, name in selected.items()},
                       "traces": {arm: executed[name][1] for arm, name in selected.items()} if replica == 0 else {}})
    return {"family": family, "seed": seed, "prefix": trace(prefix), "public_root": public_root(root),
            "selected": selected, "names": list(menu), "deltas": values, "eligible": allowed, "verification": trials}


def summarize(rows, arms):
    reports = {}
    for family in (*FAMILIES, "pooled"):
        cases = [row for row in rows if family in (row["family"], "pooled")]
        if not cases:
            continue
        trials = lambda row: [trial["outcomes"] for trial in row["verification"]]
        report = paired_statistics(cases, arms, OUTCOMES, trials)
        report["counts"] = {
            arm: {field: sum(result[arm][field] for case in cases for result in trials(case))
                  for field in ("corrected_complete", "final_wrong", "new_centered_mistake", "arrived")}
            for arm in arms}
        report["interval_counts"] = {
            arm: Counter(interval for case in cases for result in trials(case)
                         for interval in result[arm]["intervals"]) for arm in arms}
        report["interval_counts_by_bin"] = {
            arm: [Counter(result[arm]["intervals"][j] for case in cases for result in trials(case))
                  for j in range(BINS)] for arm in arms}
        reports[family] = report
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=32)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=24)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if min(args.cases, args.replicas, args.samples, args.jobs) < 1 or not 0 <= args.offset <= 90_000 - args.cases:
        parser.error("invalid case counts or seed range")
    configure(args.source)
    if any(c.evidence_mode != QUALIFIED_EVIDENCE for c in CONTROLLERS.values()):
        parser.error("prepare a qualified-evidence library first")
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(family, SEED + 100_000 * i + args.offset + j, args.samples, args.replicas)
            for i, family in enumerate(FAMILIES) for j in range(args.cases)]
    start = perf_counter()
    rows = run_parallel(paired_case, jobs, args.jobs, "qualified teacher")
    archive = args.out / "cases.jsonl.gz"
    save_archive(archive, rows)
    write_json(args.out / "summary.json", summarize(rows, ARMS) | {
        "config": config(args), "library_sha256": fingerprint(args.source / "library.json"),
        "archive_sha256": fingerprint(archive), "elapsed_s": perf_counter() - start,
    })


if __name__ == "__main__":
    main()
