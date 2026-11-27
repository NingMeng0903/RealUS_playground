"""Small development-only driver. Every unresolved episode resumes at 8/16/32.

No test split is used; a guard is an administrative censoring point, not a
task deadline or a failure label. Native trajectories and all public packets
are handled by run_adaptive.py, with its exact checkpoint state preserved.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CONFIGS = (("A_task", .30), ("A_info", .30), ("B_rollout", .30),
           ("pair_fixed", .15), ("pair_fixed", .30), ("pair_fixed", .60))


def tag(arm, dwell):
    return f"pair_fixed_dwell_{round(dwell*100):02d}" if arm == "pair_fixed" else arm


def hashes():
    names = ("acoustics.py", "adaptive_planning.py", "belief_model.py",
             "fixture.py", "native_adapter.py", "robot_motion.py", "rollout_policy.py",
             "run_adaptive.py", "run_pilot_v4.py", "train_calibrate.py")
    return {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in names}


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def latest(case):
    paths = [case / "RESULT.json"] + sorted(case.glob("RESULT_GUARD_*.json"))
    paths = [p for p in paths if p.exists()]
    return None if not paths else max(paths, key=lambda p: (
        json.loads(p.read_text()).get("fresh_count", 0), p.name))


def batches(args, arm):
    end = args.case_start + args.count
    size = args.rollout_batch_size if arm == "B_rollout" else args.batch_size
    for start in range(args.case_start, end, size):
        yield start, min(size, end-start)


def job_directory(args, arm, dwell, fixture, start, count):
    return (args.output / tag(arm, dwell) / f"fixture_{fixture:02d}" /
            f"cases_{start:04d}_{start+count-1:04d}")


def case_directory(job, arm, dwell, fixture, case_id):
    return (job / "dev" / f"fixture_{fixture:02d}_constrained" /
            tag(arm, dwell) / f"case_{case_id:04d}")


def run_job(args, arm, dwell, fixture, start, count):
    job = job_directory(args, arm, dwell, fixture, start, count)
    job.mkdir(parents=True, exist_ok=True)
    base = [sys.executable, str(HERE / "run_adaptive.py"), "--stage", "dev",
            "--index", str(fixture), "--mode", "constrained", "--arm", arm,
            "--pair-dwell", str(dwell), "--model-artifact", str(args.model),
            "--output", str(job), "--backend", "native"]
    calls_path = job / "DRIVER_CALLS.json"
    calls = json.loads(calls_path.read_text()) if calls_path.exists() else []
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")

    def invoke(extra, name):
        command = base + extra
        started = time.monotonic()
        with (job / name).open("a") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    cwd=HERE, check=False, env=environment)
        calls.append({"argv": command, "exit_code": result.returncode,
                      "wall_s": time.monotonic()-started, "log": str(job/name)})
        save(calls_path, calls)
        if result.returncode:
            print(json.dumps({"job": str(job), "exit_code": result.returncode,
                              "log": str(job/name)}), flush=True)
        return result.returncode == 0

    if not invoke(["--count", str(count), "--case-start", str(start),
                   "--max-probe-frames", "8", "--resume"], "guard08.log"):
        return {"job": str(job), "ok": False}
    for case_id in range(start, start+count):
        case = case_directory(job, arm, dwell, fixture, case_id)
        for guard in (16, 32):
            path = latest(case)
            if path is None:
                return {"job": str(job), "ok": False, "missing_case": case_id}
            result = json.loads(path.read_text())
            if not result.get("watchdog") or result.get("fresh_count", 0) >= guard:
                continue
            checkpoint = result.get("continuation_checkpoint")
            if not checkpoint:
                return {"job": str(job), "ok": False, "missing_checkpoint": str(path)}
            if not invoke(["--continue-checkpoint", checkpoint,
                           "--max-probe-frames", str(guard)],
                          f"case{case_id:04d}_guard{guard:02d}.log"):
                return {"job": str(job), "ok": False}
            continued = case / "RESULT_CONTINUED.json"
            shutil.copy2(continued, case / f"RESULT_GUARD_{guard:02d}.json")
    print(json.dumps({"finished": str(job), "cases": count}), flush=True)
    return {"job": str(job), "ok": True, "calls": len(calls)}


def summarize(args, configs):
    rows, episodes = [], {}
    for arm, dwell in configs:
        key = tag(arm, dwell)
        records = {}
        paths = []
        for fixture in args.fixtures:
            for start, count in batches(args, arm):
                job = job_directory(args, arm, dwell, fixture, start, count)
                for case_id in range(start, start+count):
                    p = latest(case_directory(job, arm, dwell, fixture, case_id))
                    if p:
                        records[(case_id, fixture)] = json.loads(p.read_text())
                        paths.append(str(p))
        n = args.count*len(args.fixtures)
        success = sum(bool(r["correct_complete"]) for r in records.values())
        false = sum(bool(r["false_complete"]) for r in records.values())
        # Abstention/no witness is also unresolved mission outcome, not proof of
        # inevitable failure; computation-incomplete cases widen the upper bound.
        unknown = n-success-false
        row = {"configuration": key, "planned_episodes": n,
               "computed_episodes": len(records), "compute_missing": n-len(records),
               "correct": success, "false_declared": false,
               "unresolved_including_compute_missing": unknown,
               "success_bounds": [success/n, (success+unknown)/n],
               "reasons": dict(Counter(r["reason"] for r in records.values())),
               "native_no_witness_count": sum(len(r.get("blacklist", [])) for r in records.values()),
               "local_option_rejections": sum(r.get("local_option_rejections", 0) for r in records.values()),
               "loss_episodes": sum(r.get("recoveryloss", 0) for r in records.values()),
               "rollout_switches": sum(d.get("reason") == "B_rollout_switch"
                                       for r in records.values() for d in r.get("decisions", [])),
               "sum_fresh_packets": sum(r.get("fresh_count", 0) for r in records.values()),
               "sum_simulated_physical_seconds": sum(r.get("simtime", 0) for r in records.values()),
               "sum_planning_wall_seconds": sum(r.get("planningwall", 0) for r in records.values()),
               "paths": paths}
        rows.append(row)
        episodes[key] = records
    comparisons = []
    if "B_rollout" in episodes:
        for other, records in episodes.items():
            if other == "B_rollout":
                continue
            common = sorted(set(records) & set(episodes["B_rollout"]))
            wins = sum(bool(episodes["B_rollout"][k]["correct_complete"]) and
                       not bool(records[k]["correct_complete"]) for k in common)
            losses = sum(not bool(episodes["B_rollout"][k]["correct_complete"]) and
                         bool(records[k]["correct_complete"]) for k in common)
            comparisons.append({"versus": other, "paired_episodes": len(common),
                                "B_observed_only_wins": wins,
                                "B_observed_only_losses": losses,
                                "warning": "Unresolved is not failure. Development pilot; mirrors share subjects."})
    out = {"scope": "v4 native continuing development pilot; no held-out efficacy claim",
           "created_utc": datetime.now(timezone.utc).isoformat(),
           "test_used": False, "no_task_deadline": True,
           "checkpoints_fresh_packets": [8, 16, 32], "rows": rows,
           "paired_descriptive_counts": comparisons}
    save(args.output / "PILOT_SUMMARY.json", out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE/"study_v4")
    parser.add_argument("--model", type=Path, default=HERE/"models/v3/MODEL.json")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--case-start", type=int, default=0)
    parser.add_argument("--fixtures", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--arms", nargs="+", default=["A_task", "A_info", "B_rollout", "pair_fixed"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rollout-batch-size", type=int, default=2)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if min(args.count, args.workers, args.batch_size, args.rollout_batch_size) < 1:
        parser.error("counts, workers and batch sizes must be positive")
    args.output = args.output.resolve()
    args.model = args.model.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    configs = [(a, d) for a, d in CONFIGS if a in args.arms]
    if not args.summarize_only:
        before = hashes()
        launch = {
            "source_sha256": before,
            "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
            "count": args.count, "case_start": args.case_start, "fixtures": args.fixtures,
            "configs": [list(c) for c in configs], "batch_size": args.batch_size,
            "rollout_batch_size": args.rollout_batch_size}
        start_path = args.output / "DRIVER_START.json"
        if start_path.exists():
            previous = json.loads(start_path.read_text())
            if any(previous.get(k) != v for k, v in launch.items()):
                parser.error("existing output has different sources, model or sample")
        else:
            save(start_path, {**launch, "created_utc": datetime.now(timezone.utc).isoformat(),
                              "workers": args.workers})
        jobs = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            schedule = sorted(configs, key=lambda c: (c[0] not in ("A_info", "B_rollout"), c[0], c[1]))
            futures = [pool.submit(run_job, args, arm, dwell, fixture, start, count)
                       for arm, dwell in schedule for start, count in batches(args, arm)
                       for fixture in args.fixtures]
            for future in as_completed(futures):
                try:
                    jobs.append(future.result())
                except Exception as exc:
                    jobs.append({"ok": False, "exception": repr(exc)})
                save(args.output/"DRIVER_JOBS.json", jobs)
        after = hashes()
        save(args.output/"DRIVER_END.json", {"source_sha256": after,
             "source_unchanged": before == after,
             "all_jobs_ok": all(j["ok"] for j in jobs)})
    result = summarize(args, configs)
    print(json.dumps([{k:v for k,v in r.items() if k != "paths"}
                      for r in result["rows"]]), flush=True)


if __name__ == "__main__":
    main()
