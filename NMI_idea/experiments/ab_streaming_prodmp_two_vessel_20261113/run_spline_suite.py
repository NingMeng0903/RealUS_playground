"""Replay all seven frozen atlas cohorts in independent disturbance slices."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from compare_spline import EXPECTED_CASES, PAIRED_ARMS, read_run

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "verification" / "DYNAMIC"


def options():
    beta2, _, _ = read_run(BASELINE / "beta2")
    dynamic = ["--motion", "ou"]
    return {
        "reference": dynamic,
        "minus-plus": [*dynamic, "--delta-beta", "-1", "1"],
        "plus-minus": [*dynamic, "--delta-beta", "1", "-1"],
        "beta2": [*dynamic, "--beta", *map(str, beta2["geometry"]["beta"])],
        "pose25": [*dynamic, "--knee-deg", "25"],
        "transient": ["--motion", "transient"],
        "static-world": ["--motion", "static", *[item for arm in PAIRED_ARMS for item in ("--arm", arm)]],
    }


def run_slice(common, cohort_options, shard, first, count):
    command = [
        *common, *cohort_options, "--out", str(shard),
        "--first-case", str(first), "--cases", str(count),
    ]
    with shard.with_suffix(".log").open("w") as log:
        subprocess.run(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, check=True)


def merge_slices(sources, out, expected):
    cases, traces, metadata = [], {}, None
    for source in sources:
        report = json.loads((source / "SUMMARY.json").read_text())
        cases.extend(report.pop("cases"))
        report.pop("case_count")
        report.pop("means")
        if metadata is None:
            metadata = report
        else:
            assert metadata == report, f"mismatched metadata for {source}"
        with np.load(source / "TRACES.npz", allow_pickle=False) as raw:
            for key in raw.files:
                if key in traces:
                    assert not key.split("_")[0].isdigit(), f"repeated case trace {key}"
                    np.testing.assert_array_equal(traces[key], raw[key])
                else:
                    traces[key] = raw[key]
    cases.sort(key=lambda row: row["case"])
    assert [row["case"] for row in cases] == list(range(expected))
    columns = ("success", "wrong", "false_complete", "coverage", "tcp_mm", "turn_deg", "duration_s", "revisions")
    means = {}
    for arm in PAIRED_ARMS:
        rows = [case["arms"][arm] for case in cases]
        means[arm] = {key: float(np.mean([row[key] for row in rows])) for key in columns}
        means[arm]["run_failures"] = sum(row["failure"] is not None for row in rows)
    metadata.update(case_count=len(cases), means=means, cases=cases)
    out.mkdir(parents=True, exist_ok=True)
    (out / "SUMMARY.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(out / "TRACES.npz", **traces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--among-us", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--spline-binding", choices=("legacy", "registered"), default="registered")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    out = args.out.resolve()
    work = out / "shards"
    work.mkdir(parents=True, exist_ok=True)
    common = [sys.executable, str(HERE / "run.py"), "atlas", "--chart", "spline",
              "--spline-binding", args.spline_binding,
              "--among-us", str(args.among_us.resolve()), "--repair", str(args.repair.resolve())]
    opts = options()
    plan = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name, n in EXPECTED_CASES.items():
            stride = {32: 11, 16: 8, 12: 6}[n]
            plan[name] = []
            for first in range(0, n, stride):
                shard = work / f"{name}-{first}"
                future = pool.submit(run_slice, common, opts[name], shard, first, min(stride, n - first))
                plan[name].append((shard, future))
        for name, tasks in plan.items():
            for shard, future in tasks:
                try:
                    future.result()
                except subprocess.CalledProcessError as error:
                    raise RuntimeError(f"read the failed slice log {shard.with_suffix('.log')}") from error
            merge_slices([shard for shard, _ in tasks], out / name, EXPECTED_CASES[name])
            print(name, "complete", flush=True)


if __name__ == "__main__":
    main()
