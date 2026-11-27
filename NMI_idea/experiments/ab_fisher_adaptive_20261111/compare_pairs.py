"""Paired confidence intervals with both fixture layouts in each object cluster."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def analyze(rows, contrasts, *, resamples=10000, seed=721505):
    methods = sorted({row["method"] for row in rows})
    cases = sorted({(row["case_id"], row["layout"]) for row in rows})
    if not cases:
        raise ValueError("no episodes")
    data = {(row["method"], row["case_id"], row["layout"]): row for row in rows}
    if len(data) != len(rows) or len(data) != len(methods) * len(cases):
        raise ValueError("duplicate or unpaired episodes")
    objects = sorted({case for case, _ in cases})
    layouts = sorted({layout for _, layout in cases})
    if len(cases) != len(objects) * len(layouts):
        raise ValueError("incomplete fixture blocks")
    metrics = (
        "success", "first_success", "wrong_window", "probes",
        "terminals", "seconds", "planning_seconds", "false_stop",
        "no_scan", "utility", "used_shift", "value_calls",
    )
    matrix = {}
    summary = {}
    for method in methods:
        arm = [data[(method, case, layout)] for case, layout in cases]
        matrix[method] = np.array([[row[k] for k in metrics] for row in arm], dtype=float)
        if not np.isfinite(matrix[method]).all():
            raise ValueError(f"non-finite metric: {method}")
        summary[method] = {
            "averages": {k: float(v) for k, v in zip(metrics, matrix[method].mean(axis=0))},
            "counts": {
                k: int(matrix[method][:, metrics.index(k)].sum())
                for k in ("success", "first_success", "wrong_window", "false_stop", "no_scan", "used_shift")
            },
            "probe_counts": dict(Counter(row["probes"] for row in arm)),
            "first_actions": dict(Counter(row["first_action"] for row in arm)),
            "first_windows": dict(Counter(row["first_window"] for row in arm)),
            "wrong_window_class": dict(Counter(row["wrong_window_class"] for row in arm)),
        }
    draws = np.random.default_rng(seed).integers(0, len(objects), size=(resamples, len(objects)))
    comparisons = {}
    for treatment, reference in contrasts:
        paired = matrix[treatment] - matrix[reference]
        clusters = paired.reshape(len(objects), len(layouts), len(metrics)).mean(axis=1)
        # Chunk resampling to avoid materializing all bootstrap arrays at once.
        boot = np.concatenate([clusters[draw].mean(axis=1) for draw in np.array_split(draws, 20)])
        limits = np.percentile(boot, [2.5, 97.5], axis=0).T
        success_col = metrics.index("success")
        value_col = metrics.index("utility")
        task_win = paired[:, success_col] > 0
        task_loss = paired[:, success_col] < 0
        action_diff = [
            i for i, (case, layout) in enumerate(cases)
            if data[treatment, case, layout]["first_action"] != data[reference, case, layout]["first_action"]
        ]
        window_diff = sum(
            data[treatment, case, layout]["first_window"] != data[reference, case, layout]["first_window"]
            for case, layout in cases
        )
        witness_ids = list(dict.fromkeys([
            *np.flatnonzero(task_win)[:4].tolist(), *np.flatnonzero(task_loss)[:4].tolist(),
            *action_diff[:4], int(np.argmax(paired[:, value_col])), int(np.argmin(paired[:, value_col])),
        ]))
        comparisons[f"{treatment} - {reference}"] = {
            "deltas": {
                k: {"mean": float(m), "ci95": ci.tolist()}
                for k, m, ci in zip(metrics, paired.mean(axis=0), limits)
            },
            "success_wins_losses": [int(task_win.sum()), int(task_loss.sum())],
            "first_action_differences": len(action_diff),
            "first_window_differences": window_diff,
            "sequence_differences": sum(
                data[treatment, case, layout]["sequence"] != data[reference, case, layout]["sequence"]
                for case, layout in cases
            ),
            "witnesses": [
                {
                    "case": cases[i],
                    "treatment": data[(treatment, *cases[i])],
                    "reference": data[(reference, *cases[i])],
                }
                for i in witness_ids
            ],
        }
    return {
        "objects": len(objects), "layouts": layouts, "records_per_arm": len(cases),
        "resamples": resamples, "seed": seed, "methods": summary, "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", type=Path, help="normalized episode rows (JSON list)")
    parser.add_argument("output", type=Path)
    parser.add_argument("--contrast", action="append", default=[], help="treatment:reference")
    args = parser.parse_args()
    pairs = [tuple(pair.split(":")) for pair in args.contrast]
    report = analyze(json.loads(args.rows.read_text()), pairs)
    with args.output.open("x") as handle:
        json.dump(report, handle, allow_nan=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
