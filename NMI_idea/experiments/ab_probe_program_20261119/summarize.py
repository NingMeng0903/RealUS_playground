"""Summaries and scene-clustered paired intervals from frozen case logs."""

from collections import Counter
import json

import numpy as np

import evaluate as e
import probe as p
from run_learned_recovery import write_json


METRICS = (e.PRIMARY, "corrected_complete", "final_wrong", "new_identity_mistake", "tcp_mm",
           "turn_degs", "seconds", "frames", "centered_complete", "new_query_image_retained",
           "feedback_revisions")
BINARY = {e.PRIMARY, "corrected_complete", "final_wrong", "new_identity_mistake", "centered_complete",
          "new_query_image_retained"}
PAIRS = (
    ("evidence", "random"), ("contrast", "random"), ("contrast", "evidence"),
    ("random", "old"), ("evidence", "old"), ("contrast", "old"), ("contrast", "mpc"),
    ("contrast", "fixed_single"), ("fixed", "mpc"),
    ("repeat", "contrast"), ("child", "repeat"), ("program", "child"), ("program", "contrast"),
    ("program", "old"), ("program", "mpc"), ("program", "fixed_single"), ("program", "fixed"),
    ("program", "same_first_fixed"), ("program", "shuffled_branch"),
    ("rollout_one", "mpc"), ("rollout_program", "rollout_one"), ("rollout_program", "program"),
    ("rollout_program", "fixed"),
)


def summaries(rows):
    n = len(rows)
    tables = {arm: np.asarray([[[t[key] for key in METRICS] for t in row["outcomes"][arm]] for row in rows])
              for arm in e.ARMS}
    boot = np.random.default_rng(18436).integers(n, size=(8000, n))
    arms = {}
    for arm, values in tables.items():
        all_trials = [trial for row in rows for trial in row["outcomes"][arm]]
        cpu = [t["decision_cpu_seconds"] for t in all_trials if t["decision_cpu_seconds"] is not None]
        initial = Counter(t["first"] for t in all_trials)
        moves = Counter((t["first"], t["second"]) for t in all_trials)
        arms[arm] = {
            "means": dict(zip(METRICS, values.mean(axis=(0, 1)).tolist(), strict=True)),
            "counts": {key: int(values[:, :, i].sum()) for i, key in enumerate(METRICS) if key in BINARY},
            "decision_cpu_ms_median": None if not cpu else 1000 * np.median(cpu),
            "decision_cpu_ms_p95": None if not cpu else 1000 * np.quantile(cpu, .95),
            "first_choices": dict(initial.most_common()), "most_common_sequences": [list(k) + [v]
                                                                                     for k, v in moves.items()],
            "probes": e.branch_counts(rows, arm) | {
                "requested_second_unavailable": sum(t["unavailable_second"] for t in all_trials),
                "chosen_handoff_with_feasible_alternative": sum(
                    t["first"] != p.HANDOFF and t["second"] == p.HANDOFF
                    and bool(t["feasible_second_alternatives"]) for t in all_trials),
            },
        }
    pairs = {}
    for method, reference in PAIRS:
        trial_diff = tables[method] - tables[reference]
        scene = trial_diff.mean(axis=1)
        bounds = np.quantile(scene[boot].mean(axis=1), [.025, .975], axis=0)
        pairs[f"{method}_minus_{reference}"] = {
            key: {"mean": float(scene[:, i].mean()), "ci95": bounds[:, i].tolist(),
                  "wins": int((trial_diff[:, :, i] > 0).sum()), "losses": int((trial_diff[:, :, i] < 0).sum())}
            for i, key in enumerate(METRICS)
        }
    return {"cases": n, "tapes_per_case": tables["mpc"].shape[1], "arms": arms, "pairs": pairs,
            "max_continuity_error": max(t["max_handoff_error"] for row in rows for trials in row["outcomes"].values()
                                        for t in trials),
            "max_tcp_mm": max(t["tcp_mm"] for row in rows for trials in row["outcomes"].values() for t in trials),
            "max_sim_seconds": max(t["seconds"] for row in rows for trials in row["outcomes"].values() for t in trials)}


def split_reports(rows):
    groups = {"all": rows}
    for field in ("family", "context"):
        for value in sorted({row[field] for row in rows}):
            groups[value] = [row for row in rows if row[field] == value]
    for family, context in p.base.CELLS:
        groups[f"{family}_{context}"] = [row for row in rows if row["family"] == family and row["context"] == context]
    return {group: summaries(subset) for group, subset in groups.items()}


def nomination_diagnostics(rows):
    sources = ("random", "evidence", "contrast")
    overlap = {}
    quality = {}
    for source in sources:
        diffs, negative, positive, spreads, costs = [], 0, 0, [], []
        for row in rows:
            labels = np.asarray(row["targets"])
            indices = [row["names"].index(n) for n in row["nominations"][source]]
            gain = labels[indices, 0]
            diffs.extend(gain.tolist())
            costs.extend(labels[indices, 3].tolist())
            negative += int(np.min(gain) < 0)
            positive += int(np.max(gain) > 0)
            spreads.append(np.max(np.r_[0, gain]) - np.min(np.r_[0, gain]))
        quality[source] = {"mean_candidate_gain": np.mean(diffs), "mean_candidate_tcp_diff": np.mean(costs),
                           "roots_with_positive_candidate": positive, "roots_with_negative_candidate": negative,
                           "mean_best_worst_gap": np.mean(spreads)}
        for other in sources:
            overlap[f"{source}/{other}"] = sum(len(set(r["nominations"][source]) & set(r["nominations"][other]))
                                                for r in rows)
    high_low = [r for r in rows if r["contrast_partner"] is not None]
    confirmed = sum(r["targets"][r["names"].index(r["nominations"]["contrast"][0])][0]
                    > r["targets"][r["names"].index(r["contrast_partner"])][0] for r in high_low)
    return {"roots": len(rows), "nomination_overlap": overlap, "teacher_quality": quality,
            "contrast_pairs_found": len(high_low), "contrast_pairs_repeated": confirmed}


def make_report(out):
    result = {"frozen_sha256": p.base.fingerprint(out / "frozen.json"),
              "test_sha256": {name: p.base.fingerprint(out / f"{name}.jsonl.gz") for name in ("test", "drift")},
              "data_diagnostics": nomination_diagnostics(p.base.read_rows(out / "first_labels.jsonl.gz")),
              "fixed": json.loads((out / "fixed.json").read_text())}
    for split in ("test", "drift"):
        result[split] = split_reports(p.base.read_rows(out / f"{split}.jsonl.gz"))
        print(split)
        for arm, values in result[split]["all"]["arms"].items():
            print(arm, values["counts"][e.PRIMARY], values["counts"]["final_wrong"],
                  round(values["means"]["tcp_mm"], 2), values["probes"])
    write_json(out / "summary.json", result)
