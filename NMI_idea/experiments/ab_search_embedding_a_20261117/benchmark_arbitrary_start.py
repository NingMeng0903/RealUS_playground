"""Paired startup and in-progress recovery from the leg atlas under image stress."""

import argparse
from collections import Counter
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
from time import process_time

import joblib
import numpy as np

from adaptive_controller import BINS, EDGES, END, FRAMES_PER_BLOCK, START, station_bin
from benchmark_window_encoding import (action_choice, choice_from_predictions, decision_inputs, draw_scene,
                                       model_inputs)
from diagnose_b_selection import run_parallel, save_archive
from learned_recovery import VISIBLE_TARGETS, fit_targets, outcome_vector, predict_deltas
from run_adaptive_recovery import paired_statistics, real_trial
from run_learned_recovery import CONTROLLERS, FAMILIES, fingerprint, read_rows, write_json
from train_startup_b import actual_values, futures, load_controllers
from verify_startup_ood import scout
from window_embedding import ViewCalibration


SPLITS = {"train": (61_000_000, 4), "test": (62_000_000, 6)}
REGIMES = ("noise", "noise_and_loss")
TASKS = ("from_start_to_end", "whole_leg_from_start", "in_progress")
ARMS = ("mpc", "frozen_b", "retrained_b")
NOISE_SCALE = 1.75
LABEL_REPEATS = 6
TEST_REPEATS = 8
LABEL_PURPOSE = 19101
TEST_PURPOSE = 19102
FIELDS = (*VISIBLE_TARGETS, "frames", "endpoint_reached", "all_task_intervals_imaged", "new_target_image",
          "centered_complete")
CALIBRATION = ViewCalibration()
OLD_MODELS = []
NEW_MODELS = []


def stress_tape(tape, row):
    outage = tape.outage.copy()
    interval = row["drop_frames"]
    if interval is not None:
        outage[slice(*interval)] = 0.
    return replace(tape, error=tape.error * NOISE_SCALE, outage=outage)


def make_row(family, regime, kind, anchor, seed):
    rng = np.random.default_rng([seed, 19110])
    blocks = 1 if kind == "cold" else anchor
    s = rng.uniform(EDGES[anchor] + .10, EDGES[anchor + 1] - .10) if kind == "cold" else START
    offset = rng.uniform(-6., 6.)
    x = float(np.clip(CONTROLLERS[family].world.nominal(s)[0] + offset, -30., 0.))
    handoff = blocks * FRAMES_PER_BLOCK
    drop = None
    if regime == "noise_and_loss":
        drop = [handoff - int(rng.integers(2, 4)) + 1, handoff + int(rng.integers(2, 5)) + 1]
    return {"family": family, "regime": regime, "kind": kind, "seed": seed, "start_s": s, "start_x": x,
            "prefix_blocks": blocks, "handoff_frame": handoff, "drop_frames": drop}


def start_case(row, task):
    controller = CONTROLLERS[row["family"]]
    scene = draw_scene(controller, row["seed"])
    scene = replace(scene, tape=stress_tape(scene.tape, row))
    if row["kind"] == "cold":
        first = station_bin(row["start_s"]) if task == "from_start_to_end" else 0
        root, prefix = scout(controller, scene, start_s=row["start_s"], start_x=row["start_x"],
                             first_required_bin=first)
    else:
        root, prefix = controller.initial_scan(scene, detailed=True, start_s=row["start_s"], start_x=row["start_x"])
        for _ in range(row["prefix_blocks"]):
            if not controller.can_continue(root):
                break
            lead = controller.choose_mpc(root, expanded=False)
            if lead is None:
                break
            controller.execute(scene, root, prefix, lead)
    return controller, scene, root, prefix


def sample_cell(job):
    family, regime, kind, anchor, seed0, quota = job
    rows = []
    for offset in range(250):
        row = make_row(family, regime, kind, anchor, seed0 + offset)
        if kind == "rolling":
            controller, _, root, prefix = start_case(row, "in_progress")
            before_loss = row["drop_frames"][0] if row["drop_frames"] is not None else row["handoff_frame"]
            tracked = any(0 in p["roles_truth"] and p["frame"] < before_loss for p in prefix.packets)
            if root.frame != row["handoff_frame"] or root.s < EDGES[1] or not tracked or not controller.can_continue(root):
                continue
        rows.append(row)
        if len(rows) == quota:
            return {"seed": seed0, "cases": rows}
    raise ValueError(f"too few eligible scans in {family} {regime} {kind} {anchor}")


def prepare(out, jobs):
    cells = [(f, r, "cold", j) for f, r, j in product(FAMILIES, REGIMES, range(BINS))]
    cells.extend((f, r, "rolling", n) for f, r, n in product(FAMILIES, REGIMES, (1, 2, 3)))
    splits, used = {}, set()
    for split, (seed, quota) in SPLITS.items():
        groups = run_parallel(sample_cell, [(*c, seed + i * 1000, quota) for i, c in enumerate(cells)],
                              jobs, f"{split} starting poses")
        rows = [r for g in groups for r in g["cases"]]
        ids = {r["seed"] for r in rows}
        if len(rows) != len(ids) or ids & used:
            raise ValueError("start-case seeds overlap")
        used.update(ids)
        archive = out / f"{split}_starts.jsonl.gz"
        save_archive(archive, rows)
        splits[split] = {"scenes": len(rows), "sha256": fingerprint(archive)}
    write_json(out / "splits.json", splits)


def task_names(row, *, training=False):
    if row["kind"] == "rolling":
        return ("in_progress",)
    if training and station_bin(row["start_s"]) == 0:
        return ("whole_leg_from_start",)
    return TASKS[:2]


def decision_row(row, task):
    controller, scene, root, prefix = start_case(row, task)
    if row["drop_frames"] and prefix.packets[-1]["y"]:
        raise ValueError("the loss burst must include the handoff frame")
    menu, inputs, common, timings = decision_inputs(controller, root, prefix, row, CALIBRATION)
    context = {"task": task, "first_required_bin": root.first_required_bin, "handoff_s": root.s,
               "handoff_x": root.x, "handoff_velocity_mm_s": [root.vs, root.vx],
               "pending_intervals": root.uncovered(), "missing_target_at_handoff": 0 not in prefix.packets[-1]["roles_truth"]}
    return controller, scene, root, prefix, menu, inputs | context, common + timings["calibrated"], common


def paired_futures(scene, row, count, purpose):
    return [replace(f, tape=stress_tape(f.tape, row)) for f in futures(scene, row["seed"], count, purpose)]


def collect_case(row):
    tasks = []
    for task in task_names(row, training=True):
        controller, scene, root, prefix, menu, inputs, _, _ = decision_row(row, task)
        bank = paired_futures(scene, row, LABEL_REPEATS, LABEL_PURPOSE)
        labels = [[outcome_vector(real_trial(controller, root, prefix, f, [a])[0], VISIBLE_TARGETS)
                   for f in bank] for a in menu.values()]
        tasks.append(inputs | {"actual_rollouts": labels})
    return {"seed": row["seed"], "tasks": tasks}


def retrained_inputs(row):
    inputs = model_inputs(row, "calibrated")
    # The task's first required interval is public and constant across this menu.
    return np.column_stack((inputs, np.full(len(inputs), row["first_required_bin"] / (BINS - 1))))


def train(out, jobs, source_digests):
    rows = run_parallel(collect_case, checked_split(out, "train"), jobs, "multi-start continuation labels")
    tasks = [t for r in rows for t in r["tasks"]]
    archive = out / "training.jsonl.gz"
    save_archive(archive, tasks)
    inputs = np.concatenate([retrained_inputs(r) for r in tasks])
    targets = np.concatenate([actual_values(r) for r in tasks])
    joblib.dump(fit_targets(inputs, targets), out / "models.joblib", compress=3)
    write_json(out / "frozen.json", {
        "source_sha256": source_digests, "training_sha256": fingerprint(archive),
        "models_sha256": fingerprint(out / "models.joblib"), "initial_scenes": len(rows),
        "task_states": len(tasks), "candidates": len(inputs), "label_rollouts": len(inputs) * LABEL_REPEATS,
        "test_starts_sha256": fingerprint(out / "test_starts.jsonl.gz"),
    })


def measured_trial(controller, root, prefix, future, action, drop_frames):
    metrics, record = real_trial(controller, root, prefix, future, [action])
    if metrics["max_handoff_error"] >= 1e-7:
        raise ValueError("non-continuous motion handoff")
    if drop_frames and any(p["y"] for p in record.packets if drop_frames[0] <= p["frame"] < drop_frames[1]):
        raise ValueError("received a packet during the programmed acquisition loss")
    sampled = {station_bin(p["s"]) for p in record.packets if 0 in p["roles_truth"]}
    post = [p for p in record.packets if p["frame"] > root.frame and 0 in p["roles_truth"]]
    return {k: metrics[k] for k in FIELDS if k in metrics} | {
        "endpoint_reached": record.motion[-1][1] >= END - .28,
        "all_task_intervals_imaged": set(range(root.first_required_bin, BINS)) <= sampled,
        "new_target_image": bool(post),
        "first_new_target_delay_s": .1 * (post[0]["frame"] - root.frame) if post else None,
        "minimum_station_mm": min(p[1] for p in record.motion),
    }


def test_case(row):
    tasks, by_first_interval = [], {}
    for task in task_names(row):
        controller, scene, root, prefix, menu, inputs, feature_cpu, mpc_cpu = decision_row(row, task)
        if root.first_required_bin in by_first_interval:
            # The two cold-start tasks are identical when the first interval is the task start.
            tasks.append(by_first_interval[root.first_required_bin] | {"task": task})
            continue
        start = process_time()
        old, old_delta = action_choice(inputs, "calibrated", OLD_MODELS)
        old_cpu = feature_cpu + process_time() - start
        start = process_time()
        new, new_delta = choice_from_predictions(inputs, predict_deltas(NEW_MODELS, retrained_inputs(inputs)))
        new_cpu = feature_cpu + process_time() - start
        choices = {"mpc": inputs["incumbent"], "frozen_b": old, "retrained_b": new}
        verification = []
        for future in paired_futures(scene, row, TEST_REPEATS, TEST_PURPOSE):
            completed = {n: measured_trial(controller, root, prefix, future, menu[n], row["drop_frames"])
                         for n in dict.fromkeys(choices.values())}
            verification.append({arm: completed[n] for arm, n in choices.items()})
        keep = {k: v for k, v in inputs.items() if k not in ("features", "margins", "calibrated_detection")}
        result = keep | {"selection": choices, "verification": verification,
                         "predicted_delta": {"frozen_b": old_delta, "retrained_b": new_delta},
                         "decision_cpu_s": {"mpc": mpc_cpu, "frozen_b": old_cpu, "retrained_b": new_cpu}}
        by_first_interval[root.first_required_bin] = result
        tasks.append(result)
    return {"seed": row["seed"], "tasks": tasks}


def count_results(rows):
    tapes = [t for r in rows for t in r["verification"]]
    f = ("corrected_complete", "final_wrong", "new_identity_mistake", "endpoint_reached",
         "all_task_intervals_imaged", "new_target_image")
    counts = {a: {k: int(sum(t[a][k] for t in tapes)) for k in f} for a in ARMS}
    loss = [t for r in rows if r["missing_target_at_handoff"] for t in r["verification"]]
    return {"scenes": len(rows), "runs_per_arm": len(tapes), "counts": counts,
            "target_missing_runs_per_arm": len(loss),
            "reacquired_after_missing": {a: sum(t[a]["new_target_image"] for t in loss) for a in ARMS}}


def identity_swap_audit():
    """Compare the public observation parameters of paired identity hypotheses."""
    audit = {}
    for family in FAMILIES:
        world = CONTROLLERS[family].world
        hypotheses = world.hypotheses
        half = len(hypotheses) // 2
        np.testing.assert_array_equal(hypotheses[:half, 1:], hypotheses[half:, 1:])
        indexes = np.arange(len(hypotheses))
        audit[family] = {}
        for region, (low, high) in {"late": (22., END), "identity_cue": (10., 18.4)}.items():
            location_gap, visibility_gap = 0., 0.
            for s, x in product(np.linspace(low, high, 25), np.linspace(-30., 0., 11)):
                means, visible = world.imaging(s, x, world.angles_deg[0], width=1.4)
                order = np.column_stack((np.lexsort((means[:, :2, 1], means[:, :2, 0]), axis=1),
                                         np.full(len(hypotheses), 2)))
                means, visible = means[indexes[:, None], order], visible[indexes[:, None], order]
                location_gap = max(location_gap, float(np.max(np.abs(means[:half] - means[half:]))))
                visibility_gap = max(visibility_gap, float(np.max(np.abs(visible[:half] - visible[half:]))))
            audit[family][region] = {"s_mm": [low, high], "max_unlabeled_centroid_delta_mm": location_gap,
                                      "max_unlabeled_visibility_delta": visibility_gap}
    return audit


def summarize(tasks):
    report = {}
    for task in TASKS:
        rows = [r for r in tasks if r["task"] == task]
        regime = {k: [r for r in rows if r["regime"] == k] for k in REGIMES}
        report[task] = count_results(rows) | {
            "by_regime": {k: count_results(v) | {"paired": paired_statistics(v, ARMS, FIELDS,
                                                                             lambda r: r["verification"])}
                          for k, v in regime.items()},
            "by_family": {f: count_results([r for r in rows if r["family"] == f]) for f in FAMILIES},
            "by_handoff_interval": [count_results([r for r in rows if station_bin(r["handoff_s"]) == j])
                                    for j in range(BINS)],
            "vs_mpc": paired_statistics(rows, ARMS, FIELDS, lambda r: r["verification"]),
            "vs_frozen": paired_statistics(rows, ("frozen_b", "retrained_b"), FIELDS, lambda r: r["verification"]),
            "decision_cpu_ms": {a: float(np.mean([r["decision_cpu_s"][a] for r in rows]) * 1000.) for a in ARMS},
            "selection_counts": {a: dict(Counter(r["selection"][a] for r in rows)) for a in ARMS},
        }
    return report


def checked_split(out, split):
    path = out / f"{split}_starts.jsonl.gz"
    config = json.loads((out / "splits.json").read_text())
    if fingerprint(path) != config[split]["sha256"]:
        raise ValueError(f"{split} start archive changed")
    return read_rows(path)


def main():
    global CALIBRATION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "test"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("results/window_encoding_v1"))
    parser.add_argument("--library", type=Path, default=Path("results/startup_b_new_library_v1"))
    parser.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    load_controllers(args.library)
    previous = json.loads((args.source / "frozen.json").read_text())
    for directory, digests in ((args.source, previous["frozen_sha256"]), (args.library, previous["source_sha256"])):
        if any(fingerprint(directory / f) != value for f, value in digests.items()):
            raise ValueError("shared motion library or frozen B changed")
    CALIBRATION = ViewCalibration(**json.loads((args.source / "calibration.json").read_text())["parameters"])
    source_digests = previous["source_sha256"] | previous["frozen_sha256"]
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        prepare(args.out, args.jobs)
        return
    if args.stage == "train":
        train(args.out, args.jobs, source_digests)
        return
    frozen = json.loads((args.out / "frozen.json").read_text())
    for f in ("models.joblib", "test_starts.jsonl.gz"):
        key = "models_sha256" if f == "models.joblib" else "test_starts_sha256"
        if fingerprint(args.out / f) != frozen[key]:
            raise ValueError(f"frozen {f} changed")
    if source_digests != frozen["source_sha256"]:
        raise ValueError("shared inputs changed after training")
    OLD_MODELS[:] = joblib.load(args.source / "models.joblib")["calibrated"]
    NEW_MODELS[:] = joblib.load(args.out / "models.joblib")
    rows = run_parallel(test_case, checked_split(args.out, "test"), args.jobs, "paired arbitrary-start closed loops")
    tasks = [t for r in rows for t in r["tasks"]]
    archive = args.out / "test.jsonl.gz"
    save_archive(archive, tasks)
    write_json(args.out / "summary.json", {"test_sha256": fingerprint(archive), "independent_test_seeds": len(rows),
                                           "tasks": summarize(tasks), "identity_swap_audit": identity_swap_audit()})


if __name__ == "__main__":
    main()
