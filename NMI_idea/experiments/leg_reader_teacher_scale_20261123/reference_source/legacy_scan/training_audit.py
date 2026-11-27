"""Compare frozen CT readers and test the train-time slow recheck at test time."""
from argparse import ArgumentParser
from dataclasses import replace
import gzip
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

import student_transfer as study
from exit_continuation import feasible_by_station, metrics
from geometry import TASKS, digest


def load(directory):
    model = directory / "ct_only.joblib"
    metadata = json.loads((directory / "ct_only_training.json").read_text())
    if digest(model) != metadata["model_sha256"] or not metadata["no_blender_geometry_loaded"]:
        raise ValueError(f"Unexpected training freeze in {directory}")
    return joblib.load(model)


def use(bundle):
    study.HISTORY = bundle["history"]
    study.HISTORY_NEIGHBORS = bundle.get("history_neighbors", False)
    study.RISK_THRESHOLD = bundle["thresholds"]["witness"]


def ct_run(job):
    spec, model, trial = study.structured_job(job)
    end, _, _ = study.execute(spec, trial, model)
    out = {}
    for name, reader in BUNDLES.items():
        use(reader)
        out[f"mpc_{name}_reader"] = study.report(end.reader)
        if name != "original":
            student, _, _ = study.execute(spec, trial, model, reader["aggregated"])
            out[f"{name}_student_slow"] = study.report(student.reader)
    return out


def paired_replay(record):
    task, profile, tape = (record[key] for key in ("task", "profile", "tape"))
    spec = study.test_definition(task)
    seed = study.env.TEST_SEED + 1000 * list(TASKS).index(task) + tape
    trial = study.env.Trial(3, seed, profile)
    model = study.activate(0, 0, 8)
    use(BUNDLES["scaled"])
    mpc, _, _ = study.execute(spec, trial, model)
    out = {}
    for name, bundle in BUNDLES.items():
        use(bundle)
        out[f"mpc_{name}_reader"] = metrics(mpc, spec, FEASIBLE[task])
        if name != "original":
            for slow in (False, True):
                end, _, _ = study.execute(spec, trial, model, bundle["aggregated"], permit_slow=slow)
                label = f"{name}_student_slow" if slow else f"{name}_student"
                out[label] = metrics(end, spec, FEASIBLE[task])
    for label, key in (("mpc_scaled_reader", "mpc"), ("scaled_student", "search_student_history")):
        for field, value in record["arms"][key].items():
            assert out[label][field] == value, (task, profile, tape, label, field)
    return {"task": task, "profile": profile, "tape": tape, "arms": out}


def totals(results):
    values = {}
    for arm in results[0]:
        items = [row[arm] for row in results]
        values[arm] = {
            "cases": len(items),
            "captured_all_six": sum(item["captured_regions"] == 6 for item in items),
            **{field: sum(item[field] for item in items)
               for field in ("correct_regions", "wrong_regions", "wrong_delivery", "pass")},
        }
        if "time_s" in items[0]:
            values[arm].update(
                mean_seconds=float(np.mean([item["time_s"] for item in items])),
                mean_stations_seen=float(np.mean([item["unique_target_stations"] for item in items])),
                accessible_25mm_coverage=sum(item["clear_fine_bins_success"] for item in items),
                rechecks=sum(item["rechecks"] for item in items),
                slow_rechecks=sum(item["slow_segments"] for item in items),
            )
    return values


def main(root, workers):
    global BUNDLES, FEASIBLE
    small = root / "ct_student_transfer"
    scaled = root / "ct_student_scaling"
    context = root / "ct_student_neighbor_context"
    BUNDLES = {"original": load(small), "scaled": load(scaled), "neighbor": load(context)}
    study.setup_train(scaled / "ct_smplx_shapes.npz")
    # The audit seeds come from none of the training or calibration phases.
    cases = [replace(job, seed=job.seed + 1_100_000)
             for job in study.training_jobs("ct_only", "validation", 4)]
    ct = study.grouped(ct_run, cases, workers)

    study.setup_test(root)
    FEASIBLE = {task: feasible_by_station(study.test_definition(task)) for task in TASKS}
    with gzip.open(scaled / "ct_only_blender_trials.jsonl.gz", "rt") as stream:
        original = [json.loads(line) for line in stream]
    rows = study.grouped(paired_replay, original, workers)
    long = [row for row in rows if row["task"] == "groin_to_distal_calf"]
    result = {
        "ct": {"replays": len(ct), "new_noise_seeds": len({job.seed for job in cases}),
               "independent_CT_subjects": 1, "results": totals(ct)},
        "blender": {"all": totals([r["arms"] for r in rows]),
                    "730mm": totals([r["arms"] for r in long]),
                    "paired_slow_vs_scaled": {
                        field: study.env.bootstrap(rows, "scaled_student_slow", "scaled_student", field)
                        for field in ("six_bins_success", "correct_regions", "wrong_delivery", "time_s")
                    },
                    "paired_slow_with_neighbor_context": {
                        field: study.env.bootstrap(rows, "neighbor_student_slow", "neighbor_student", field)
                        for field in ("six_bins_success", "correct_regions", "wrong_delivery", "time_s")
                    }},
    }
    (context / "paired_training_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        main(args.root, args.workers)
