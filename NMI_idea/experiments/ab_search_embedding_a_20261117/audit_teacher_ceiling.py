"""Check absolute teacher completion and the ceiling of the shared first-action menu."""

import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from adaptive_controller import BINS, CENTER_MM, END, IDENTITY_ACCEPT, station_bin
from budgeted_recovery import CAPS
from diagnose_b_selection import VERIFY_PURPOSE, run_parallel, save_archive
from learned_recovery import OBJECTIVE, TARGETS, outcome_vector
from run_adaptive_recovery import paired_statistics, real_trial, shared_prefix, tape_for
from run_learned_recovery import CONTROLLERS, FAMILIES, configure, fingerprint, read_rows, write_json


ARMS = ("mpc", "teacher", "privileged", "privileged_cost")
MASK = (1 << BINS) - 1
CALIBRATION_PURPOSE = 2411


def frame_coverage(record):
    """Attribute outcome loss along a frozen executed path, using evaluator truth."""
    centered = observed = accepted = 0
    last_good = {}
    for packet in record.packets:
        j = station_bin(packet["s"])
        if j < 0 or abs(packet["target_truth"][0] - packet["x"]) > CENTER_MM:
            continue
        bit = 1 << j
        centered |= bit
        detected = [i for i, (role, y) in enumerate(zip(packet["roles_truth"], packet["y"], strict=True))
                    if role == 0 and abs(y[0]) <= CENTER_MM]
        if detected:
            observed |= bit
        if packet["accepted_slot"] in detected:
            accepted |= bit
            last_good[j] = packet["frame"]
    revoked = {(event["witness"], event["bin"]) for event in record.retractions}
    retained = {event["bin"]: event for event in record.certificates
                if (event["frame"], event["bin"]) not in revoked}
    stale = [j for j, event in retained.items() if event["identity_ok"] and not event["center_ok"]
             and last_good.get(j, -1) > event["frame"]]
    return centered, observed, accepted, stale


def rescore_public(controller, record):
    """Reselect an image per bin from already accepted packets, on a frozen path."""
    belief = np.asarray(record.packets[-1]["belief"])
    preferred = {}
    for packet in record.packets:
        slot = packet["accepted_slot"]
        j = station_bin(packet["s"])
        if j < 0 or slot < 0 or abs(packet["y"][slot][0]) > CENTER_MM:
            continue
        _, conditional = controller.world.likelihood(np.asarray(packet["y"]), packet["s"], packet["x"], packet["q"])
        identity = conditional[:, slot]
        if belief @ identity < IDENTITY_ACCEPT:
            continue
        nominal = controller.world.target_grid(packet["s"], packet["q"])[:, 0]
        quality = float(belief @ (identity * (abs(nominal - packet["x"]) <= CENTER_MM)))
        if j not in preferred or quality > preferred[j][0]:
            preferred[j] = quality, packet
    # Evaluation labels are read only after all replacements have been chosen.
    wrong = any(p["roles_truth"][p["accepted_slot"]] != 0 for _, p in preferred.values())
    good = sum(p["roles_truth"][p["accepted_slot"]] == 0 and abs(p["target_truth"][0] - p["x"]) <= CENTER_MM
               for _, p in preferred.values())
    return {"corrected_complete": record.motion[-1][1] >= END - .28 and good == BINS, "final_wrong": wrong}


def replay(controller, root, prefix, scenario, action, diagnostics, *, rescore=False):
    metrics, record = real_trial(controller, root, prefix, scenario, [action], detailed=rescore)
    result = {field: metrics[field] for field in TARGETS}
    if diagnostics:
        centered, observed, accepted, stale = frame_coverage(record)
        reached = bool(record.motion[-1][1] >= END - .28)
        result |= {"reached": reached, "centered_bins": centered, "raw_bins": observed, "accepted_bins": accepted,
                   "stale_off_center_bins": stale, "good_bins": metrics["good_bins"]}
        if metrics["corrected_complete"] and (accepted != MASK or not reached):
            raise ValueError("correct completion lacks its required accepted image")
        if rescore:
            result["public_postprocessing"] = rescore_public(controller, record)
    return result


def within_caps(deltas):
    eligible = np.ones(len(deltas), dtype=bool)
    for name, cap in CAPS.items():
        eligible &= deltas[:, TARGETS.index(name)] <= cap + 1e-8
    return eligible


def select_privileged(names, estimates, incumbent, eligible):
    """Select with the hidden geometry and independent calibration disturbances."""
    i = max(np.flatnonzero(eligible), key=lambda k: (estimates[k, 0], estimates[k] @ OBJECTIVE,
                                                    names[k] == incumbent))
    return names[i]


def audit_case(job):
    case, samples = job
    family, seed = case["family"], case["seed"]
    controller = CONTROLLERS[family]
    scene = controller.world.draw(seed)
    root, prefix = shared_prefix(controller, scene)
    menu = controller.options_shared(root, renderer="fit")
    names = list(menu)
    incumbent, teacher = (case["selected"][arm] for arm in ("mpc", "teacher"))
    if names != case["names"] or not names:
        raise ValueError("the candidate menu disagrees with the archived study")
    np.testing.assert_allclose(root.belief, case["public_root"]["belief"], atol=1e-12, rtol=0.)

    calibration = [tape_for(scene.latent, seed, i, CALIBRATION_PURPOSE) for i in range(samples)]
    estimates = np.array([[outcome_vector(replay(controller, root, prefix, future, action, False))
                           for future in calibration] for action in menu.values()]).mean(axis=1)
    mpc_i = names.index(incumbent)
    suitable = within_caps(estimates - estimates[mpc_i])
    choices = {"mpc": incumbent, "teacher": teacher,
               "privileged": select_privileged(names, estimates, incumbent, np.ones(len(names), dtype=bool)),
               "privileged_cost": select_privileged(names, estimates, incumbent, suitable)}

    rows = []
    for saved in case["verification"]:
        future = tape_for(scene.latent, seed, saved["replica"], VERIFY_PURPOSE)
        all_replays = [replay(controller, root, prefix, future, action, True,
                              rescore=name in (incumbent, teacher)) for name, action in menu.items()]
        observed = np.array([outcome_vector(result) for result in all_replays])
        for arm in ("mpc", "teacher"):
            np.testing.assert_allclose(observed[names.index(choices[arm])], outcome_vector(saved["outcomes"][arm]),
                                       atol=1e-12, rtol=0.)
        rows.append({"replica": saved["replica"], "actions": all_replays,
                     "outcomes": {arm: all_replays[names.index(name)] for arm, name in choices.items()},
                     "ceilings": {
                         "same_tail_hard_budget": bool(np.any(observed[:, 0])),
                         "same_tail_paired_cost_caps": bool(np.any((observed[:, 0] > 0.) &
                                                                  within_caps(observed - observed[mpc_i]))),
                         "raw_image_and_arrival": any(v["reached"] and v["raw_bins"] == MASK for v in all_replays),
                     }})
    return {"family": family, "seed": seed, "names": names, "selected": choices,
            "calibration_means": estimates, "calibration_cost_eligible": suitable, "verification": rows}


def stage_counts(outcomes):
    stages = Counter()
    for out in outcomes:
        if out["corrected_complete"]:
            stages["complete"] += 1
        elif not out["reached"]:
            stages["endpoint_not_reached"] += 1
        elif out["raw_bins"] != MASK:
            stages["missing_centered_raw_image"] += 1
            stages["never_physically_centered_in_some_bin"] += int(out["centered_bins"] != MASK)
        elif out["accepted_bins"] != MASK:
            stages["missing_correct_accepted_image"] += 1
        else:
            stages["valid_accepted_images_not_retained"] += 1
            stages["later_valid_after_retained_off_center"] += int(bool(out["stale_off_center_bins"]))
    return stages


def summarize(rows):
    report = {}
    for family in (*FAMILIES, "pooled"):
        cases = [row for row in rows if row["family"] == family or family == "pooled"]
        paired = paired_statistics(cases, ARMS, TARGETS, lambda case: [v["outcomes"] for v in case["verification"]])
        trials = [v for case in cases for v in case["verification"]]
        paired["pairs_per_arm"] = len(trials)
        paired["counts"] = {arm: int(sum(v["outcomes"][arm]["corrected_complete"] for v in trials)) for arm in ARMS}
        paired["ceilings"] = {k: sum(v["ceilings"][k] for v in trials) for k in trials[0]["ceilings"]}
        paired["forensics"] = {}
        for arm in ("mpc", "teacher"):
            outcomes = [v["outcomes"][arm] for v in trials]
            paired["forensics"][arm] = {
                "failure_breakdown": stage_counts(outcomes),
                "all_raw_and_reached": sum(v["reached"] and v["raw_bins"] == MASK for v in outcomes),
                "all_accepted_and_reached": sum(v["reached"] and v["accepted_bins"] == MASK for v in outcomes),
                "public_reselection_correct": sum(v["public_postprocessing"]["corrected_complete"] for v in outcomes),
                "public_reselection_wrong": sum(v["public_postprocessing"]["final_wrong"] for v in outcomes),
                "public_reselection_wins": sum(v["public_postprocessing"]["corrected_complete"]
                                               and not v["corrected_complete"] for v in outcomes),
                "public_reselection_losses": sum(v["corrected_complete"]
                                                 and not v["public_postprocessing"]["corrected_complete"] for v in outcomes),
            }
        report[family] = paired
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calibration", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()
    if min(args.calibration, args.jobs) <= 0:
        parser.error("positive calibration and worker counts required")
    configure(args.source)
    args.out.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    rows = run_parallel(audit_case, [(row, args.calibration) for row in read_rows(args.cases)], args.jobs,
                        "teacher ceiling")
    save_archive(args.out / "ceiling_cases.jsonl.gz", rows)
    write_json(args.out / "ceiling_summary.json", summarize(rows) | {
        "case_archive_sha256": fingerprint(args.cases), "calibration_tapes": args.calibration,
        "elapsed_s": perf_counter() - started,
    })


if __name__ == "__main__":
    main()
