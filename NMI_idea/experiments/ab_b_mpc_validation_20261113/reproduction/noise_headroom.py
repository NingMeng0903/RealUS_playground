"""Noise-separated development diagnostic for first-action headroom.

The frozen prefix and common probe are retained exactly.  For each case and
replicate, only the independent suffix acquisition innovations are resampled.
The physical-time dropout interval, persistent stress ghost, case anatomy,
motion durations, and preflight action graph stay fixed.  The same random
innovation bank is reused across views at a stage, which gives a paired
comparison of first roots under common exogenous noise.

This is a diagnostic experiment.  A root selected from the first replicate
half is evaluated on the second half, and the oracle selection is never used
by the archived B or MPC controller.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
import sys

# Keep BLAS from multiplying each process.  This must happen before importing
# numpy through the helper module.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import lzma
import numpy as np


HERE = Path(__file__).resolve().parent
LEGACY = HERE / "legacy_headroom.py"
STUDY = HERE / "RealUS_playground" / "NMI_idea" / "experiments" / "ab_tro_y_continuation_20261112"
HOLDOUT = STUDY / "holdout"
EXECUTOR = None


def load_common(n):
    with lzma.open(HOLDOUT / "COMMON.jsonl.xz", "rt") as stream:
        return [json.loads(line) for _, line in zip(range(n), stream)]


@dataclass
class NoiseBank:
    """Common random numbers for all views at one physical suffix stage."""

    detect: np.ndarray
    scale: np.ndarray
    normal: np.ndarray
    clutter_branch: float
    clutter_branch_mode: float
    clutter_normal: np.ndarray
    clutter_uniform: np.ndarray

    @classmethod
    def make(cls, case_seed, replicate, stage):
        rng = np.random.default_rng(np.random.SeedSequence([int(case_seed), 50517, int(replicate), int(stage)]))
        return cls(
            detect=rng.random(2),
            scale=rng.random(2),
            normal=rng.normal(size=(2, 2)),
            clutter_branch=float(rng.random()),
            clutter_branch_mode=float(rng.random()),
            clutter_normal=rng.normal(size=2),
            clutter_uniform=rng.random(2),
        )


def suffix_observe(world, case, t, site, view, bank):
    """MapWorld.observe with only suffix acoustic innovations resampled.

    Pose packets use the original deterministic case/tick stream.  Target
    detection, target noise, and non-persistent clutter use the replicate bank.
    Persistent clutter and physical-time target dropout remain case-fixed.
    """

    from knee_world import Frame, DEPTH, WIDTH

    t = float(t)
    tick = round(t * 1000)
    pose_rng = np.random.default_rng(np.random.SeedSequence([case.seed, 147, tick]))
    q, speed = case.true_angle_rate(t)
    reading_q = q + (case.pose_bias + pose_rng.normal(0.0, 0.10) if world.noisy_pose else 0.0)
    reading_v = speed + (pose_rng.normal(0.0, 0.15) if world.noisy_pose else 0.0)

    base, rows = world.projection(q, site, view)
    offset = np.array([case.w_subject_mm[site], case.branch_offset_mm, case.z_mm(t), case.w_local])
    expected = base + rows @ offset
    values, labels = [], []
    pd = world.detection_probability(q, site, view) + 0.15 - case.dropout
    target_miss = (
        world.mode == "stress"
        and case._miss_start_s <= t <= case._miss_end_s
        and world.identity_site(site)
    )
    for label, center in enumerate(expected):
        if target_miss and label == case._miss_label:
            continue
        if bank.detect[label] >= pd:
            continue
        scale = 3.0 if bank.scale[label] < 0.075 else 1.0
        observed = center + bank.normal[label] * np.array([1.0 * scale, 0.68])
        if abs(observed[0]) <= WIDTH and DEPTH[0] <= observed[1] <= DEPTH[1]:
            values.append(observed)
            labels.append(label)

    persistent = world._persistent_clutter(case, q, site, view, base)
    if persistent is not None:
        values.append(persistent)
        labels.append(-1)
    elif bank.clutter_branch < case.clutter * (1.0 if world.identity_site(site) else 0.5):
        if world.identity_site(site) and bank.clutter_branch_mode < 0.78:
            center = base[case.ghost_label] + bank.clutter_normal * np.array([2.9, 1.35])
            center = np.clip(center, [-WIDTH, DEPTH[0]], [WIDTH, DEPTH[1]])
        else:
            center = np.array([-WIDTH, DEPTH[0]]) + bank.clutter_uniform * np.array([2.0 * WIDTH, DEPTH[1] - DEPTH[0]])
        values.append(center)
        labels.append(-1)
    if values:
        order = np.argsort(np.asarray(values)[:, 0], kind="stable")
        values = np.asarray(values)[order]
        labels = tuple(labels[i] for i in order)
    else:
        values, labels = np.zeros((0, 2)), ()
    return Frame(float(t), float(reading_q), float(reading_v), site, view, values, labels)


def frame_public(frame):
    return {
        "site": frame.site,
        "view": frame.view,
        "t": frame.t,
        "q": frame.q,
        "velocity": frame.velocity,
        "centroids_mm": frame.centroids.tolist(),
    }


def worker_init():
    global EXECUTOR
    sys.path.insert(0, str(HERE))
    import legacy_headroom  # noqa: E402

    EXECUTOR = legacy_headroom.import_frozen_executor()


def evaluate_pair(case, start_state, first_path, tail_path, first_frame, terminal_frame, reserve):
    state = EXECUTOR.update(start_state, first_frame)
    state = EXECUTOR.update(state, terminal_frame)
    claim = EXECUTOR.target_claim(state, 30)
    outcome = EXECUTOR.evaluate_claim(claim, terminal_frame, True)
    return {
        "strict": bool(outcome["strict"]),
        "correct_fresh": bool(outcome["correct_fresh"]),
        "false_confirm": bool(outcome["false_confirm"]),
        "target_visible": bool(1 in terminal_frame.labels),
        "confidence": float(claim["confidence"]),
        "label": outcome["label_for_evaluation"],
        "tail_view": int(tail_path.goal[1]),
        "first_view": int(first_path.goal[1]),
        "reserve": reserve,
        "first_t": float(first_frame.t),
        "terminal_t": float(terminal_frame.t),
    }


def run_case(payload):
    common, replicates = payload
    executor = EXECUTOR
    if executor is None:
        raise RuntimeError("worker was not initialized")
    seed = int(common["seed"])
    case = executor.WORLD.case(seed)
    prefix = executor.prepare(executor.WORLD, case)
    if executor.prefix_hash(prefix) != common["prefix_sha"]:
        raise AssertionError(f"prefix hash mismatch for case {common['index']}")
    probe_t = float(common["probe_motion"]["t_end"])
    probe = executor.WORLD.observe(case, probe_t, 3, 3)
    expected_probe = common["common_probe_frame"]
    observed_probe = frame_public(probe)
    if any(observed_probe[key] != expected_probe[key] for key in ("site", "view", "t", "q", "velocity")):
        raise AssertionError(f"common probe clock/pose mismatch for case {common['index']}")
    if observed_probe["centroids_mm"] != expected_probe["centroids_mm"]:
        raise AssertionError(f"common probe mismatch for case {common['index']}")
    start_state = executor.update(prefix, probe)
    first_paths, tail_graph = executor.legal_graph(executor.RESERVES_MM["wide"])
    first_by_view = {int(path.goal[1]): path for path in first_paths}
    roots = tuple(sorted(first_by_view))
    result = {str(root): [] for root in roots}
    for replicate in range(replicates):
        first_bank = NoiseBank.make(seed, replicate, 1)
        terminal_bank = NoiseBank.make(seed, replicate, 2)
        for root in roots:
            first_path = first_by_view[root]
            first_t = probe_t + float(first_path.duration)
            first_frame = suffix_observe(executor.WORLD, case, first_t, *first_path.goal, first_bank)
            first_state = executor.update(start_state, first_frame)
            tail_path, tail_scores = executor.choose_real_tail(
                executor.WORLD, first_state.belief, tail_graph[root]
            )
            terminal_t = first_t + float(tail_path.duration)
            terminal_frame = suffix_observe(executor.WORLD, case, terminal_t, *tail_path.goal, terminal_bank)
            row = evaluate_pair(case, start_state, first_path, tail_path, first_frame, terminal_frame, "wide")
            row["replicate"] = replicate
            row["tail_scores"] = [float(value) for value in tail_scores["values"]]
            result[str(root)].append(row)
    return {"index": int(common["index"]), "seed": seed, "roots": roots, "replicates": result}


def rate(rows, key):
    return float(np.mean([bool(row[key]) for row in rows])) if rows else 0.0


def select_root(rows_by_root):
    scored = {int(root): rate(rows, "strict") for root, rows in rows_by_root.items()}
    # Prefer center on exact training ties so the diagnostic never invents a
    # side preference from a tie.
    best = max(scored, key=lambda root: (scored[root], int(root == 3), -abs(root - 3)))
    return int(best), scored


def summarize(cases, replicates):
    roots = [2, 3, 4]
    pooled = {str(root): [] for root in roots}
    selected_test_rows = []
    per_case = []
    for case in cases:
        split = {
            str(root): {
                "train": case["replicates"][str(root)][: replicates // 2],
                "test": case["replicates"][str(root)][replicates // 2 :],
            }
            for root in roots
        }
        train_by_root = {root: split[str(root)]["train"] for root in roots}
        best_root, train_rates = select_root(train_by_root)
        test_rates = {root: rate(split[str(root)]["test"], "strict") for root in roots}
        test_center = test_rates[3]
        selected_test = test_rates[best_root]
        train_center = train_rates[3]
        per_case.append({
            "index": case["index"],
            "best_root_train": best_root,
            "train_rates": {str(root): train_rates[root] for root in roots},
            "test_rates": {str(root): test_rates[root] for root in roots},
            "train_advantage": float(train_rates[best_root] - train_center),
            "test_advantage": float(selected_test - test_center),
            "stable_positive": bool(best_root != 3 and selected_test > test_center),
            "noise_dependent": bool(best_root != 3 and selected_test <= test_center),
            "best_root_test": best_root,
        })
        selected_test_rows.extend(split[str(best_root)]["test"])
        for root in roots:
            pooled[str(root)].extend(case["replicates"][str(root)])

    pooled_rates = {
        str(root): {
            "n": len(pooled[str(root)]),
            "strict_rate": rate(pooled[str(root)], "strict"),
            "correct_fresh_rate": rate(pooled[str(root)], "correct_fresh"),
            "false_confirm_rate": rate(pooled[str(root)], "false_confirm"),
            "target_visible_rate": rate(pooled[str(root)], "target_visible"),
        }
        for root in roots
    }
    center_test_rows = []
    for case in cases:
        center_test_rows.extend(case["replicates"]["3"][replicates // 2 :])
    selected_rate = rate(selected_test_rows, "strict")
    center_rate = rate(center_test_rows, "strict")
    return {
        "pooled_forced_root": pooled_rates,
        "cross_fitted_per_case": {
            "n_case_test_rows": len(selected_test_rows),
            "selected_strict_rate": selected_rate,
            "center_strict_rate": center_rate,
            "strict_advantage": selected_rate - center_rate,
            "cases_selected_side_root": sum(row["best_root_train"] != 3 for row in per_case),
            "cases_stable_positive": sum(row["stable_positive"] for row in per_case),
            "cases_noise_dependent": sum(row["noise_dependent"] for row in per_case),
            "case_rows": per_case,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--replicates", type=int, default=16)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--out", type=Path, default=HERE / "noise_headroom")
    args = parser.parse_args()
    if args.n <= 0 or args.n > 128:
        raise ValueError("--n must be in 1..128")
    if args.replicates < 2 or args.replicates % 2:
        raise ValueError("--replicates must be an even number >= 2")
    args.out.mkdir(parents=True, exist_ok=True)
    common = load_common(args.n)
    payloads = [(row, args.replicates) for row in common]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("fork"), initializer=worker_init) as pool:
        cases = list(pool.map(run_case, payloads))
    output = {
        "schema": "moving_y_suffix_noise_headroom_v1",
        "source_commit": "cdb1705a02f60382ffbe9e42f2ed420e27a28ddb",
        "n_cases": len(cases),
        "replicates_per_root_per_case": args.replicates,
        "reserve": "wide",
        "split": {"train_replicates": list(range(args.replicates // 2)), "test_replicates": list(range(args.replicates // 2, args.replicates))},
        "same_prefix_and_common_probe": True,
        "same_case_dropout_and_persistent_ghost": True,
        "suffix_noise_only": True,
        "common_random_numbers_across_views": True,
        "public_filter_and_choose_real_tail": True,
        "diagnostic_only": True,
        "limitations": [
            "Root selection uses strict outcomes on the training half and is evaluated only on the disjoint test half.",
            "No new policy is trained and no evaluator label is exposed to the public filter or choose_real_tail.",
            "The acoustic innovation bank is a local common-random-number simulator. It does not claim that the synthetic noise distribution matches a physical scanner.",
            "Motion legality is inherited from the frozen wide-reserve preflight graph. Robot execution is not rerun by this diagnostic.",
        ],
        "summary": summarize(cases, args.replicates),
        "cases": cases,
    }
    out_file = args.out / f"NOISE_HEADROOM_{args.n}x{args.replicates}.json"
    out_file.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    compact = {
        "schema": output["schema"],
        "n_cases": output["n_cases"],
        "replicates_per_root_per_case": output["replicates_per_root_per_case"],
        "summary": output["summary"],
        "output": str(out_file),
    }
    (args.out / f"SUMMARY_{args.n}x{args.replicates}.json").write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
