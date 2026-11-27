#!/usr/bin/env python3
"""Recompute checks from archived outcomes only; never runs the simulator or trains."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
REF = BASE / "reference_results"
OUT = HERE / "results" / "archive_audit.json"
BOOTSTRAPS = 20_000
BOOTSTRAP_SEED = 20261120


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise AssertionError(f"empty archive: {path}")
    for r in rows:
        for k in ("seed", "policy", "correct", "wrong"):
            if k not in r:
                raise AssertionError(f"missing {k} in {path}")
        r["seed"] = int(r["seed"])
        r["correct"] = int(r["correct"])
        r["wrong"] = int(r["wrong"])
    return rows


def assert_paired_grid(rows, label):
    seen = set()
    arms = sorted({r["policy"] for r in rows})
    grid = {(r["seed"], r["noise"]) for r in rows}
    for r in rows:
        key = (r["seed"], r["noise"], r["policy"])
        if key in seen:
            raise AssertionError(f"duplicate seed/noise/policy key in {label}: {key}")
        seen.add(key)
    expected = {(seed, noise, arm) for seed, noise in grid for arm in arms}
    if seen != expected:
        missing, extra = expected-seen, seen-expected
        raise AssertionError(f"incomplete paired grid in {label}; missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}")
    return {"unique_seed_noise_policy": True, "complete_same_seed_noise_grid_for_all_arms": True,
            "arms": arms, "seed_noise_cells": len(grid), "seed_clusters": len({r["seed"] for r in rows})}


def summarize(rows):
    policies = sorted({r["policy"] for r in rows})
    output = {}
    for p in policies:
        group = [r for r in rows if r["policy"] == p]
        output[p] = {"n": len(group), "correct": sum(r["correct"] for r in group),
                     "wrong": sum(r["wrong"] for r in group),
                     "ever": None, "strict_at_cine": None,
                     "unavailable_csv_columns": ["ever", "strict_at_cine"]}
    return output


def clustered_ci(rows, a, b):
    """Paired difference in correctness; resample seed clusters, preserving rows/noises."""
    keys = sorted({r["seed"] for r in rows})
    by = {(r["seed"], r["noise"], r["policy"]): r["correct"] for r in rows}
    pairs = []
    for seed in keys:
        # Each seed contributes the average of its paired noise-level differences.
        diffs = []
        for noise in sorted({r["noise"] for r in rows if r["seed"] == seed}):
            va, vb = by.get((seed, noise, a)), by.get((seed, noise, b))
            if va is not None and vb is not None:
                diffs.append(va - vb)
        if diffs:
            pairs.append((seed, float(np.mean(diffs))))
    if not pairs:
        return None
    values = np.asarray([d for _, d in pairs], dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    boot_means = values[draws].mean(axis=1) * 100.0
    pair_diffs = []
    for seed in keys:
        for noise in sorted({r["noise"] for r in rows if r["seed"] == seed}):
            va, vb = by.get((seed, noise, a)), by.get((seed, noise, b))
            if va is not None and vb is not None:
                pair_diffs.append(va - vb)
    wins = sum(x > 0 for x in pair_diffs)
    losses = sum(x < 0 for x in pair_diffs)
    ties = len(pair_diffs) - wins - losses
    return {"comparison": f"{a}_minus_{b}", "difference_pp": float(values.mean() * 100),
            "clustered_bootstrap_95ci_pp": [float(x) for x in np.percentile(boot_means, [2.5, 97.5])],
            "seed_clusters": len(pairs), "paired_seed_noise_rows": len(pair_diffs),
            "wins": wins, "losses": losses, "ties": ties,
            "algorithm": f"Paired mean correctness difference; resample {len(pairs)} unique seed clusters with replacement, retain all noise rows within each selected seed; {BOOTSTRAPS} replicates; NumPy PCG64 seed {BOOTSTRAP_SEED}; percentile interval."}


def count_checks(csv_arms, summary, summary_key, policy_map):
    checks = []
    for csv_policy, json_policy in policy_map.items():
        if csv_policy not in csv_arms:
            raise AssertionError(f"CSV missing arm {csv_policy}")
        c = csv_arms[csv_policy]
        expected = summary[json_policy]
        check = {"csv_policy": csv_policy, "summary_arm": json_policy,
                 "csv_n_equals_summary_n": c["n"] == expected["n"],
                 "csv_correct_equals_summary_correct": c["correct"] == expected["correct"],
                 "csv_wrong_equals_summary_wrong": c["wrong"] == expected["wrong"]}
        if not all(v for k, v in check.items() if k.startswith("csv_")):
            raise AssertionError(f"archived count mismatch: {check}")
        checks.append(check)
    return {"summary_file": summary_key, "checks": checks, "all_passed": True}


def failure_decomposition(arms):
    result = {}
    for p, a in arms.items():
        if a.get("ever") is None or a.get("strict_at_cine") is None:
            continue
        n, e, s, d = a["n"], a["ever"], a["strict_at_cine"], a["correct"]
        assert 0 <= d <= s <= e <= n, f"nested events violated for {p}"
        result[p] = {"never_strict": n-e, "strict_only_in_transit_or_prefix": e-s,
                     "strict_at_cine_but_not_correctly_delivered": s-d,
                     "correct_delivery": d, "partition_total": n,
                     "nested_event_note": "D ⊆ S ⊆ E; the categories are disjoint differences: ¬E, E\\S, S\\D, D."}
    return result


def source_audit():
    engine = BASE / "work/gallbladder_feedback/engine.py"
    policy = BASE / "work/gallbladder_feedback/policy.py"
    train = BASE / "work/gallbladder_feedback/train.py"
    runtime = BASE / "simulator/gallbladder_runtime.py"
    asset_manifest = json.loads((REF / "asset_inputs.json").read_text(encoding="utf-8"))
    assets = {}
    for env, spec in asset_manifest.items():
        path = Path(os.environ[env]).expanduser().resolve() if os.environ.get(env) else BASE / "local_assets" / spec["default_filename"]
        assets[env] = {"path": str(path), "exists": path.is_file(), "sha256_manifest": spec["sha256"]}
    return {
        "engine.run_station_visible_gate": {"file": str(engine.relative_to(BASE)), "record": "run_station() schedules feedback checks after QUERY_START at QUERY_EVERY intervals, with remaining-motion and deadline checks; it also requires a visible_frame in the recent history window before constructing intervention options."},
        "policy.KINDS": {"file": str(policy.relative_to(BASE)), "record": "KINDS enumerates continue, keep_view, center, fan_left, fan_right, and recall."},
        "train.ExtraTrees": {"file": str(train.relative_to(BASE)), "record": "Imports sklearn.ensemble.ExtraTreesRegressor and fits it during local policy training; archived audit does not import or train it."},
        "psdp_first_station_gate": {"file": str((BASE / 'work/gallbladder_feedback/evaluate.py').relative_to(BASE)), "record": "evaluate.test defines the PSDP gate callable so index != 0 immediately returns True; the learned gate therefore only decides whether to enable feedback at the first proposed station. Later stations use feedback without a learned gate decision."},
        "asset_readiness": {"runtime_source": str(runtime.relative_to(BASE)), "manifest": str((REF / "asset_inputs.json").relative_to(BASE)), "assets": assets,
                            "all_ready": all(x["exists"] for x in assets.values())}
    }


def main():
    primary_rows = read_csv(REF / "paired_primary.csv")
    replication_rows = read_csv(REF / "paired_replication.csv")
    primary = summarize(primary_rows)
    primary["hand"] = dict(primary["hand_servo"])
    replication = summarize(replication_rows)
    heldout = json.loads((REF / "heldout_summary.json").read_text(encoding="utf-8"))
    residual = json.loads((REF / "residual_holdout_summary.json").read_text(encoding="utf-8"))
    primary_grid = assert_paired_grid(primary_rows, "paired_primary.csv")
    replication_grid = assert_paired_grid(replication_rows, "paired_replication.csv")
    checks = [count_checks(primary, heldout["arms"], "heldout_summary.json", {
        "stationary": "stationary", "multimodal": "multimodal", "hand_servo": "hand_servo",
        "reverse_only": "reverse_only", "student_feedback": "student_feedback", "joint": "joint", "psdp": "psdp"}),
        count_checks(replication, residual["arms"], "residual_holdout_summary.json", {
            "stationary": "stationary", "hand": "hand", "residual": "residual", "zero_margin": "zero_margin"})]
    # Attach strict-frame counts from archived summaries, clearly distinguished from CSV recomputation.
    for p, jp in {"stationary":"stationary", "multimodal":"multimodal", "hand":"hand_servo", "hand_servo":"hand_servo", "reverse_only":"reverse_only", "student_feedback":"student_feedback", "joint":"joint", "psdp":"psdp"}.items():
        primary[p]["ever"] = heldout["arms"][jp]["ever"]
        primary[p]["strict_at_cine"] = heldout["arms"][jp]["strict_at_cine"]
        primary[p]["ever_strict_source"] = "heldout_summary.json (not present in CSV; not independently recomputed)"
    comparisons = [clustered_ci(primary_rows, "hand_servo", "stationary"),
                   clustered_ci(primary_rows, "student_feedback", "hand_servo"),
                   clustered_ci(primary_rows, "reverse_only", "hand_servo"),
                   clustered_ci(primary_rows, "joint", "hand_servo"),
                   clustered_ci(primary_rows, "psdp", "hand_servo"),
                   clustered_ci(replication_rows, "hand", "stationary"),
                   clustered_ci(replication_rows, "residual", "hand")]
    learn_hand = {"primary_student_feedback_vs_hand": comparisons[1],
                  "primary_reverse_only_vs_hand": comparisons[2],
                  "primary_joint_vs_hand": comparisons[3],
                  "primary_psdp_vs_hand": comparisons[4],
                  "replication_residual_vs_hand": comparisons[6],
                  "pairwise_wins": {name: {"learning_wins": item["wins"], "handwritten_wins": item["losses"], "ties": item["ties"]}
                                    for name, item in (("student_feedback", comparisons[1]), ("reverse_only", comparisons[2]),
                                                       ("joint", comparisons[3]), ("psdp", comparisons[4]),
                                                       ("replication_residual", comparisons[6]))}}
    report = {"audit_type": "archived_outcomes_only",
              "scope": "CSV/JSON parsing and deterministic recalculation only; no simulation, no training, no model imports.",
              "datasets": {"paired_primary.csv": {"rows": len(primary_rows), "grid_assertions": primary_grid, "arms": primary},
                           "paired_replication.csv": {"rows": len(replication_rows), "grid_assertions": replication_grid, "arms": replication}},
              "summary_count_consistency": checks,
              "strict_failure_decomposition": {"source": "archived heldout_summary.json E/S/D counts; CSV lacks ever and strict_at_cine columns", "primary": failure_decomposition(primary)},
              "clustered_bootstrap_comparisons": comparisons,
              "learning_vs_handwritten_win_counts": learn_hand,
              "source_audit": source_audit()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"audit_type": report["audit_type"], "primary": {p: {k: a[k] for k in ("n", "correct", "wrong", "ever", "strict_at_cine")} for p, a in primary.items()},
                      "summary_count_checks_passed": len(checks),
                      "learning_vs_handwritten": learn_hand["pairwise_wins"], "output": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
