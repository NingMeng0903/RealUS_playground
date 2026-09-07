#!/usr/bin/env python3
"""Offline CSV timing audit. No robot, viewer, or controller process is opened."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np


def analyze(path: Path, period_ms: float) -> dict:
    fields = {
        "dt_actual_s": 1000.0, "tick_inner_ms": 1.0, "tick_send_ms": 1.0,
        "tick_log_ms": 1.0, "qpik_total_ms": 1.0, "qpik_assembly_ms": 1.0,
        "qpik_qp1_solve_ms": 1.0, "qpik_qp2_solve_ms": 1.0,
        "feedback_age_s": 1000.0,
    }
    values = {key: [] for key in fields}
    reasons, qp1, qp2 = Counter(), Counter(), Counter()
    rows, missed = 0, 0
    active_cbf = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        # New native stage columns are picked up without interpreting missing
        # columns in old CSVs as zero cost.
        for key in reader.fieldnames or []:
            if key.startswith("qpik_") and key.endswith("_ms"):
                fields[key] = 1.0
                values[key] = []
        for row in reader:
            rows += 1
            reasons[row.get("qpik_fallback_reason", "")] += 1
            qp1[row.get("qpik_qp1_status", "")] += 1
            qp2[row.get("qpik_qp2_status", "")] += 1
            for key, factor in fields.items():
                try:
                    value = float(row[key]) * factor
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    values[key].append(value)
            try:
                missed += float(row["deadline_slack_s"]) < 0.0
                active_cbf.append(int(row["n_cbf"]))
            except (KeyError, TypeError, ValueError):
                pass
    timing = {}
    for key, samples in values.items():
        if samples:
            data = np.asarray(samples)
            timing[key] = dict(zip(("p50_ms", "p95_ms", "p99_ms", "max_ms"),
                                  np.percentile(data, [50, 95, 99, 100]).tolist()))
            timing[key]["samples"] = len(samples)
            timing[key]["over_period"] = int(np.sum(data > period_ms))
    return {"path": str(path), "rows": rows, "period_ms": period_ms,
            "negative_deadline_slack_rows": int(missed),
            "fallback_reasons": dict(reasons), "qp1_status": dict(qp1),
            "qp2_status": dict(qp2), "max_active_cbf": max(active_cbf, default=None),
            "timing": timing}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--period-ms", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps([analyze(path, args.period_ms) for path in args.csv],
                         indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
