"""Replay startup probes and isolate the motion that recovers the first interval."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from adaptive_controller import END, FRAMES_PER_BLOCK, station_bin
from controller import PROGRESS_SPEED
from diagnose_b_selection import run_parallel, save_archive
from evaluate_startup_without_gates import FRESH_PURPOSE, recorded_menu
from recovery_world import FOV_MAX, FOV_MIN, NOISE_SD, PACKET_OUTAGE
from run_adaptive_recovery import paired_statistics, real_trial
from run_learned_recovery import fingerprint, read_rows, write_json
from train_startup_b import TEST_TAPES, futures, load_controllers, menu_metadata, start_case
from verify_startup_ood import OUTCOMES


PAIRED = ("fixed", "completion_first", "no_gates")
ABLATIONS = ("flat", "no_lateral_endpoint", "advance_faster", "mirror", "wait")
ARMS = (*PAIRED, *ABLATIONS)
FIELDS = (
    "corrected_complete", "first_interval_recaptured", "first_block_target_in_first_interval",
    "frames_in_first_interval_during_first_block", "expected_target_frames_in_first_interval_during_first_block",
    "correct_first_interval_witness_from_first_block", "early_witness_confirmed_later",
    "final_wrong", "new_identity_mistake", "tcp_mm", "turn_degs", "seconds",
)
DIRECT_PLAN_PHASE = .5


def make_probe(controller, root, name, request):
    for renderer in ("fit", "waypoint"):
        candidate = controller.free_candidate(root, name, *request, renderer=renderer)
        if candidate is not None:
            return candidate
    raise ValueError(f"ablation {name} has no admissible trajectory")


def action_set(controller, root, row):
    menu = controller.options_shared(root, renderer="fit")
    if menu_metadata(menu) != recorded_menu(row):
        raise ValueError("recorded mechanics and live admissible menu differ")
    selected = {arm: menu[row["selection"][arm]] for arm in PAIRED}
    ds, dx, bow = selected["fixed"].request
    recipes = {
        "flat": (ds, dx, 0.),
        "no_lateral_endpoint": (ds, 0., bow),
        "advance_faster": (PROGRESS_SPEED, dx, bow),
        "mirror": (ds, -dx, -bow),
    }
    probes = {name: make_probe(controller, root, name, request) for name, request in recipes.items()}
    return selected | probes | {"wait": menu["wait"]}


def target_detection_prob(world, scene, packet):
    means, visibility = world.imaging(
        packet["s"], packet["x"] - scene.lateral_jitter, packet["q"], width=scene.shadow_width_factor,
        lane_shift=scene.shadow_shift, depth_shift=scene.depth_jitter,
    )
    mean = means[scene.latent, 0]
    inside = float(np.prod(ndtr((FOV_MAX - mean) / NOISE_SD) - ndtr((FOV_MIN - mean) / NOISE_SD)))
    return float((1. - PACKET_OUTAGE) * visibility[scene.latent, 0] * inside)


def summarize_run(controller, scene, root, outcome, record):
    packets = record.packets
    post = [p for p in packets if p["frame"] > root.frame]
    first_end = root.frame + FRAMES_PER_BLOCK
    first = [p for p in post if p["frame"] <= first_end]
    in_first_interval = [p for p in post if station_bin(p["s"]) == 0]
    early = [p for p in in_first_interval if p["frame"] <= first_end]
    target = [p["frame"] for p in in_first_interval if 0 in p["roles_truth"]]
    early_prob = [target_detection_prob(controller.world, scene, p) for p in early]
    ledger = packets[-1]["visible_ledger"]
    by_frame = {p["frame"]: p for p in packets}
    correct = [j for j, w in enumerate(ledger) if w is not None and by_frame[w[0]]["roles_truth"][w[1]] == 0]
    first_card = ledger[0]
    early_witness = 0 in correct and root.frame < first_card[0] <= first_end
    accepted = None
    if early_witness:
        accepted = min(e["accepted_at"] for e in record.certificates if e["bin"] == 0
                       and e["frame"] == first_card[0] and e["slot"] == first_card[1])
    return {k: outcome[k] for k in OUTCOMES if k in outcome} | {
        "first_interval_recaptured": bool(target),
        "first_block_target_in_first_interval": any(p <= first_end for p in target),
        "frames_in_first_interval_during_first_block": len(early),
        "expected_target_frames_in_first_interval_during_first_block": sum(early_prob),
        "correct_first_interval_witness_from_first_block": early_witness,
        "early_witness_confirmed_later": accepted is not None and accepted > first_end,
        "first_target_frame": target[0] if target else None,
        "first_exit_frame": next((p["frame"] for p in post if station_bin(p["s"]) > 0), None),
        "first_block_end_s": first[-1]["s"],
        "first_block_end_x": first[-1]["x"],
        "final_s": packets[-1]["s"],
        "end_reached": packets[-1]["s"] >= END - .28,
        "unfilled_intervals": [j for j, w in enumerate(ledger) if w is None],
        "wrong_intervals": [j for j, w in enumerate(ledger) if w is not None and j not in correct],
    }


def trajectory_record(controller, scene, root, record):
    return [{"frame": p["frame"], "s": p["s"], "x": p["x"], "target_detected": 0 in p["roles_truth"],
             "target_visibility": target_detection_prob(controller.world, scene, p)}
            for p in record.packets if root.frame < p["frame"] <= root.frame + FRAMES_PER_BLOCK]


def replay_case(row):
    controller, scene, root, prefix = start_case(row)
    probes = action_set(controller, root, row)
    outcomes, failures = [], []
    for tape, future in enumerate(futures(scene, row["seed"], TEST_TAPES, FRESH_PURPOSE)):
        trials, cache = {}, {}
        diagnostic = any(not row["verification"][tape][arm]["corrected_complete"] for arm in PAIRED[1:])
        details = {}
        for arm, action in probes.items():
            if action.name not in cache:
                result, record = real_trial(controller, root, prefix, future, [action], detailed=True)
                if result["max_handoff_error"] >= 1e-7:
                    raise ValueError("motion handoff was discontinuous")
                cache[action.name] = summarize_run(controller, future, root, result, record)
                if diagnostic and arm in PAIRED:
                    details[action.name] = {
                        "first_block": trajectory_record(controller, future, root, record),
                        "decision_path": [d["action"] for d in record.decisions],
                    }
            trials[arm] = cache[action.name]
            if arm in PAIRED:
                archived = row["verification"][tape][arm]
                for field in FIELDS:
                    if field in archived and trials[arm][field] != archived[field]:
                        raise ValueError(f"paired replay changed {arm} {field} at case {row['seed']}")
        if diagnostic:
            failures.append({"tape": tape, "traces": {arm: details[probes[arm].name] for arm in PAIRED}})
        outcomes.append(trials)
    plans = {}
    for arm, action in probes.items():
        phase_index = round((len(action.s) - 1) * DIRECT_PLAN_PHASE)
        plans[arm] = {
            "name": action.name, "request": getattr(action, "request", None),
            "renderer": getattr(action, "renderer", "legacy_prodmp"),
            "nominal_mid_lateral_mm": float(action.lateral.x[phase_index] - root.x),
            "nominal_end_s_mm": float(action.s[-1]),
        }
    return {k: row[k] for k in ("seed", "family", "group")} | {
        "root_s": root.s, "root_x": root.x, "root_frame": root.frame, "nominal_plans": plans,
        "verification": outcomes, "paired_failure_traces": failures,
    }


def summarize(rows):
    selected = ("corrected_complete", "first_interval_recaptured", "first_block_target_in_first_interval",
                "correct_first_interval_witness_from_first_block", "early_witness_confirmed_later")
    stats = paired_statistics(rows, ARMS, FIELDS, lambda r: r["verification"])
    stats["counts"] = {arm: {k: int(sum(t[arm][k] for r in rows for t in r["verification"]))
                             for k in selected} for arm in ARMS}
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/startup_b_without_gates_v1"))
    parser.add_argument("--controller", type=Path, default=Path("results/startup_b_new_library_v1"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    load_controllers(args.controller)
    summary = json.loads((args.source / "fresh_summary.json").read_text())
    archive = args.source / "fresh.jsonl.gz"
    if fingerprint(archive) != summary["archive_sha256"]:
        raise ValueError("fresh results changed before the mechanism audit")
    rows = run_parallel(replay_case, read_rows(archive), args.jobs, "startup probe mechanism")
    args.out.mkdir(parents=True, exist_ok=True)
    save_archive(args.out / "replay.jsonl.gz", rows)
    write_json(args.out / "summary.json", {"source_archive_sha256": fingerprint(archive), "all": summarize(rows)})


if __name__ == "__main__":
    main()
